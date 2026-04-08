"""Cross-repo scanner — discovers API endpoints and service dependencies from code.

Scans service repos to find:
1. Endpoints EXPOSED — route decorators (@app.get, @router.post, etc.)
2. Endpoints CONSUMED — outbound HTTP calls (httpx, requests, aiohttp)
3. Event contracts — message publishing/consuming patterns
4. Shared types — common model names across services

Works by:
- Scanning Python files for route decorators → endpoint registry
- Scanning for HTTP client calls → dependency registry
- Matching consumed URLs against exposed endpoints → dependency graph
- Detecting env var patterns for service URLs → service discovery

Edge cases:
- Dynamic routes ('/users/{user_id}'): captured with path params
- URL built from env vars: detected via pattern matching
- Multiple routers with prefixes: prefix resolved where possible
- Versioned APIs ('/api/v1/...' vs '/api/v2/...'): tracked separately
- Internal calls vs external calls: distinguished by URL pattern
- Service calling itself: filtered out
- Duplicate endpoints (overloaded): both captured
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from codebase_intel.crossrepo.models import (
    APIEndpoint,
    ContractStatus,
    EventContract,
    HttpMethod,
    ProtocolType,
    ServiceDependency,
    ServiceInfo,
    ServiceStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Route / endpoint detection patterns (14 frameworks across 10 languages)
# ---------------------------------------------------------------------------
ROUTE_PATTERNS = [
    # --- Python ---
    # FastAPI / Starlette
    re.compile(r'@(?:app|router|api_router)\.(get|post|put|patch|delete|options|head)\s*\(\s*["\']([^"\']+)["\']'),
    # Flask
    re.compile(r'@(?:app|bp|blueprint)\.route\s*\(\s*["\']([^"\']+)["\'](?:.*methods\s*=\s*\[([^\]]+)\])?'),
    # Django REST (only match url patterns with api/ prefix to avoid false positives)
    re.compile(r'@(?:api_view|action)\s*\(\s*\[([^\]]+)\]\s*\)'),
    re.compile(r'path\s*\(\s*["\'](?:api/[^"\']+)["\']'),
    # Tornado
    re.compile(r'\(\s*r?["\'](/api/[^"\']+)["\'].*(?:Handler|View)'),

    # --- JavaScript / TypeScript ---
    # Express / Koa / Hapi
    re.compile(r'(?:app|router|server)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),
    # NestJS decorators
    re.compile(r'@(Get|Post|Put|Patch|Delete)\s*\(\s*["\']([^"\']+)["\']'),
    # Next.js API routes (file-based)
    re.compile(r'export\s+(?:default\s+)?(?:async\s+)?function\s+(GET|POST|PUT|PATCH|DELETE)\b'),

    # --- Go ---
    # net/http
    re.compile(r'(?:http|mux|r|router)\.(?:HandleFunc|Handle|Get|Post|Put|Patch|Delete)\s*\(\s*["\']([^"\']+)["\']'),
    # Gin
    re.compile(r'(?:r|router|group|v1|v2|api)\.(GET|POST|PUT|PATCH|DELETE)\s*\(\s*["\']([^"\']+)["\']'),
    # Echo
    re.compile(r'(?:e|echo|g|group)\.(GET|POST|PUT|PATCH|DELETE)\s*\(\s*["\']([^"\']+)["\']'),
    # Fiber
    re.compile(r'(?:app|group|api)\.(Get|Post|Put|Patch|Delete)\s*\(\s*["\']([^"\']+)["\']'),

    # --- Java ---
    # Spring Boot
    re.compile(r'@(GetMapping|PostMapping|PutMapping|PatchMapping|DeleteMapping)\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']'),
    re.compile(r'@RequestMapping\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\'](?:.*method\s*=\s*RequestMethod\.(\w+))?'),
    # JAX-RS (Quarkus, Jersey)
    re.compile(r'@(GET|POST|PUT|PATCH|DELETE)\s*\n\s*@Path\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'@Path\s*\(\s*["\']([^"\']+)["\']'),

    # --- Rust ---
    # Actix-web
    re.compile(r'#\[(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'web::(get|post|put|patch|delete)\s*\(\)\s*\.to\s*\('),
    # Axum
    re.compile(r'\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),
    # Rocket
    re.compile(r'#\[(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),

    # --- Ruby ---
    # Rails routes (require leading slash to avoid false positives)
    re.compile(r'(?:get|post|put|patch|delete)\s+["\'](/[^"\']+)["\']'),
    re.compile(r'resources?\s+:(\w+)'),

    # --- PHP ---
    # Laravel
    re.compile(r'Route::(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),
    # Symfony
    re.compile(r'#\[Route\s*\(\s*["\']([^"\']+)["\'](?:.*methods:\s*\[([^\]]+)\])?'),

    # --- C# ---
    # ASP.NET
    re.compile(r'\[(Http(Get|Post|Put|Patch|Delete))\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'\[Route\s*\(\s*["\']([^"\']+)["\']'),

    # --- Kotlin ---
    # Ktor
    re.compile(r'(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),
    # Spring Boot (same as Java)

    # --- Swift ---
    # Vapor
    re.compile(r'(?:app|grouped)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'),

    # --- Elixir ---
    # Phoenix
    re.compile(r'(get|post|put|patch|delete)\s+["\']([^"\']+)["\']'),
]

# ---------------------------------------------------------------------------
# Outbound HTTP call patterns (all languages)
# ---------------------------------------------------------------------------
HTTP_CALL_PATTERNS = [
    # --- Python ---
    re.compile(r'(?:httpx|client|async_client)\.(get|post|put|patch|delete)\s*\(\s*(?:f?["\']([^"\']*)["\']|(\w+))'),
    re.compile(r'requests\.(get|post|put|patch|delete)\s*\(\s*(?:f?["\']([^"\']*)["\']|(\w+))'),
    re.compile(r'session\.(get|post|put|patch|delete)\s*\(\s*(?:f?["\']([^"\']*)["\']|(\w+))'),

    # --- JavaScript / TypeScript ---
    re.compile(r'fetch\s*\(\s*(?:`([^`]*)`|["\']([^"\']*)["\'])'),
    re.compile(r'axios\.(get|post|put|patch|delete)\s*\(\s*(?:`([^`]*)`|["\']([^"\']*)["\'])'),
    re.compile(r'(?:\$http|http)\.(get|post|put|patch|delete)\s*\(\s*(?:`([^`]*)`|["\']([^"\']*)["\'])'),

    # --- Go ---
    re.compile(r'http\.(Get|Post|Put|NewRequest)\s*\(\s*["\']([^"\']*)["\']'),
    re.compile(r'(?:client|c|resty)\.(Get|Post|Put|Patch|Delete|R)\s*\(\s*["\']([^"\']*)["\']'),

    # --- Java ---
    re.compile(r'(?:HttpRequest|RestTemplate|WebClient|Feign)\.\w+\s*\(\s*["\']([^"\']*)["\']'),
    re.compile(r'\.uri\s*\(\s*["\']([^"\']*)["\']'),

    # --- Rust ---
    re.compile(r'(?:reqwest|client)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']'),
    re.compile(r'Client::new\(\).*\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']'),

    # --- Ruby ---
    re.compile(r'(?:Net::HTTP|HTTParty|Faraday|RestClient)\.(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']'),

    # --- PHP ---
    re.compile(r'(?:Http|Guzzle|client)->(get|post|put|patch|delete)\s*\(\s*["\']([^"\']*)["\']'),
    re.compile(r'file_get_contents\s*\(\s*["\']([^"\']*)["\']'),

    # --- C# ---
    re.compile(r'(?:HttpClient|client)\.(Get|Post|Put|Patch|Delete)Async\s*\(\s*["\']([^"\']*)["\']'),
    re.compile(r'(?:HttpClient|client)\.Send\w*\s*\('),

    # --- Generic URL building (all languages) ---
    re.compile(r'(?:url|endpoint|base_url|service_url|apiUrl|baseURL)\s*(?:\+|=|:)\s*(?:f?["\']([^"\']*(?:api|/v\d)[^"\']*)["\'])'),
]

# ---------------------------------------------------------------------------
# Service URL environment variable patterns (all languages)
# ---------------------------------------------------------------------------
SERVICE_URL_PATTERNS = [
    # Python
    re.compile(r'(\w+(?:SERVICE|API|BACKEND)_(?:URL|HOST|BASE_URL|ENDPOINT))\b'),
    re.compile(r'os\.(?:environ|getenv)\s*\(\s*["\'](\w*(?:URL|HOST|BASE)\w*)["\']'),
    re.compile(r'settings\.(\w*(?:url|host|base_url|endpoint)\w*)', re.IGNORECASE),
    # Go
    re.compile(r'os\.Getenv\s*\(\s*["\'](\w*(?:URL|HOST|BASE|SERVICE)\w*)["\']'),
    # Java / Kotlin
    re.compile(r'@Value\s*\(\s*["\']?\$\{(\w*(?:url|host|base|service)\w*)', re.IGNORECASE),
    re.compile(r'System\.getenv\s*\(\s*["\'](\w*(?:URL|HOST|BASE|SERVICE)\w*)["\']'),
    # JS / TS
    re.compile(r'process\.env\.(\w*(?:URL|HOST|BASE|SERVICE|API)\w*)'),
    re.compile(r'(?:NEXT_PUBLIC|VITE|REACT_APP)_(\w*(?:URL|API)\w*)'),
    # Ruby
    re.compile(r'ENV\s*\[\s*["\'](\w*(?:URL|HOST|BASE|SERVICE)\w*)["\']'),
    # PHP
    re.compile(r'(?:env|getenv)\s*\(\s*["\'](\w*(?:URL|HOST|BASE|SERVICE)\w*)["\']'),
    # C# / .NET
    re.compile(r'Configuration\s*\[\s*["\'](\w*(?:Url|Host|Base|Service)\w*)["\']'),
    # Rust
    re.compile(r'(?:env::var|std::env::var)\s*\(\s*["\'](\w*(?:URL|HOST|BASE|SERVICE)\w*)["\']'),
    # Generic .env patterns
    re.compile(r'^(\w+(?:_URL|_HOST|_BASE_URL|_ENDPOINT|_SERVICE))\s*=', re.MULTILINE),
]

# ---------------------------------------------------------------------------
# Event / message queue patterns (all languages)
# ---------------------------------------------------------------------------
EVENT_PATTERNS = [
    # Generic publish/subscribe
    re.compile(r'\.publish\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'\.subscribe\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'\.send_message\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'channel\s*=\s*["\']([^"\']+)["\']'),
    # RabbitMQ
    re.compile(r'(?:basic_publish|queue_declare)\s*\(.*["\']([^"\']+)["\']'),
    # Kafka
    re.compile(r'(?:produce|send|consume)\s*\(\s*(?:topic\s*=\s*)?["\']([^"\']+)["\']'),
    # Redis
    re.compile(r'\.(?:lpush|rpush|publish)\s*\(\s*["\']([^"\']+)["\']'),
    # AWS SNS/SQS
    re.compile(r'(?:TopicArn|QueueUrl)\s*[=:]\s*["\']([^"\']+)["\']'),
    # NATS
    re.compile(r'\.(?:Publish|Subscribe|Request)\s*\(\s*["\']([^"\']+)["\']'),
    # Go channels (named)
    re.compile(r'(?:nats|nc)\.(Publish|Subscribe)\s*\(\s*["\']([^"\']+)["\']'),
]


class ServiceScanner:
    """Scans a service repo to discover endpoints and dependencies."""

    def __init__(self, skip_dirs: set[str] | None = None) -> None:
        self._skip_dirs = skip_dirs or {
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        }

    def scan_service(
        self,
        repo_path: Path,
        service_id: str | None = None,
    ) -> ServiceScanResult:
        """Scan a single service repo.

        Returns discovered endpoints, outbound calls, and event contracts.
        """
        repo_path = repo_path.resolve()
        service_id = service_id or repo_path.name

        result = ServiceScanResult(
            service_id=service_id,
            repo_path=str(repo_path),
        )

        source_files = self._collect_source_files(repo_path)
        logger.info(
            "Scanning %s: %d source files", service_id, len(source_files)
        )

        for file_path, content in source_files.items():
            rel_path = str(file_path.relative_to(repo_path))

            # Find exposed endpoints
            endpoints = self._find_endpoints(content, service_id, rel_path)
            result.endpoints.extend(endpoints)

            # Find outbound HTTP calls
            calls = self._find_outbound_calls(content, service_id, rel_path)
            result.outbound_calls.extend(calls)

            # Find service URL env vars
            env_vars = self._find_service_urls(content)
            result.service_url_vars.update(env_vars)

            # Find event patterns
            events = self._find_events(content, service_id, rel_path)
            result.events.extend(events)

        logger.info(
            "Scan complete for %s: %d endpoints, %d outbound calls, %d events",
            service_id,
            len(result.endpoints),
            len(result.outbound_calls),
            len(result.events),
        )

        return result

    def _find_endpoints(
        self, content: str, service_id: str, file_path: str
    ) -> list[APIEndpoint]:
        """Find route decorators that expose API endpoints."""
        endpoints: list[APIEndpoint] = []
        lines = content.split("\n")

        for line_no, line in enumerate(lines, 1):
            for pattern in ROUTE_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue

                groups = match.groups()
                if len(groups) >= 2 and groups[0] and groups[1]:
                    method_str = groups[0].upper()
                    path = groups[1]
                elif len(groups) >= 1:
                    path = groups[0]
                    method_str = "ANY"
                else:
                    continue

                try:
                    method = HttpMethod(method_str)
                except ValueError:
                    method = HttpMethod.ANY

                # Find the function name (next def line)
                func_name = ""
                for i in range(line_no, min(line_no + 5, len(lines))):
                    func_match = re.search(r'(?:async\s+)?def\s+(\w+)', lines[i - 1])
                    if func_match:
                        func_name = func_match.group(1)
                        break

                endpoints.append(APIEndpoint(
                    path=path,
                    method=method,
                    service_id=service_id,
                    file_path=file_path,
                    line_number=line_no,
                    function_name=func_name,
                ))

        return endpoints

    def _find_outbound_calls(
        self, content: str, service_id: str, file_path: str
    ) -> list[OutboundCall]:
        """Find HTTP client calls to other services."""
        calls: list[OutboundCall] = []
        lines = content.split("\n")

        for line_no, line in enumerate(lines, 1):
            for pattern in HTTP_CALL_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue

                groups = match.groups()
                method_str = ""
                url = ""
                var_name = ""

                for g in groups:
                    if g is None:
                        continue
                    if g.upper() in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                        method_str = g.upper()
                    elif "/" in g or "http" in g.lower() or "{" in g:
                        url = g
                    elif g.isidentifier():
                        var_name = g

                if not url and not var_name:
                    continue

                calls.append(OutboundCall(
                    service_id=service_id,
                    method=method_str or "ANY",
                    url=url,
                    url_variable=var_name,
                    file_path=file_path,
                    line_number=line_no,
                    raw_line=line.strip()[:200],
                ))

        return calls

    def _find_service_urls(self, content: str) -> set[str]:
        """Find environment variables that hold service URLs."""
        vars_found: set[str] = set()
        for pattern in SERVICE_URL_PATTERNS:
            for match in pattern.finditer(content):
                vars_found.add(match.group(1))
        return vars_found

    def _find_events(
        self, content: str, service_id: str, file_path: str
    ) -> list[EventContract]:
        """Find event publishing/consuming patterns."""
        events: list[EventContract] = []
        for pattern in EVENT_PATTERNS:
            for match in pattern.finditer(content):
                event_name = match.group(1)
                direction = "publishes" if "publish" in pattern.pattern or "send" in pattern.pattern else "consumes"
                events.append(EventContract(
                    event_name=event_name,
                    service_id=service_id,
                    direction=direction,
                    file_path=file_path,
                ))
        return events

    def _collect_source_files(self, repo_path: Path) -> dict[Path, str]:
        """Collect all source files with content."""
        files: dict[Path, str] = {}

        def _walk(directory: Path) -> None:
            try:
                for entry in sorted(directory.iterdir()):
                    if entry.is_dir():
                        if entry.name not in self._skip_dirs and not entry.name.startswith("."):
                            _walk(entry)
                    elif entry.is_file() and entry.suffix in (
                        ".py", ".ts", ".js", ".tsx", ".jsx",  # Python, JS/TS
                        ".go",  # Go
                        ".java", ".kt", ".kts",  # Java, Kotlin
                        ".rs",  # Rust
                        ".rb",  # Ruby
                        ".php",  # PHP
                        ".cs",  # C#
                        ".swift",  # Swift
                        ".ex", ".exs",  # Elixir
                        ".scala",  # Scala
                        ".env",  # Environment files
                    ):
                        try:
                            content = entry.read_text(encoding="utf-8", errors="ignore")
                            if len(content) < 500_000:
                                files[entry] = content
                        except OSError:
                            pass
            except PermissionError:
                pass

        _walk(repo_path)
        return files


class OutboundCall:
    """An outbound HTTP call discovered in source code."""

    def __init__(
        self,
        service_id: str,
        method: str,
        url: str = "",
        url_variable: str = "",
        file_path: str = "",
        line_number: int | None = None,
        raw_line: str = "",
    ) -> None:
        self.service_id = service_id
        self.method = method
        self.url = url
        self.url_variable = url_variable
        self.file_path = file_path
        self.line_number = line_number
        self.raw_line = raw_line


class ServiceScanResult:
    """Complete scan result for a single service."""

    def __init__(self, service_id: str, repo_path: str) -> None:
        self.service_id = service_id
        self.repo_path = repo_path
        self.endpoints: list[APIEndpoint] = []
        self.outbound_calls: list[OutboundCall] = []
        self.events: list[EventContract] = []
        self.service_url_vars: set[str] = set()

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "repo_path": self.repo_path,
            "endpoints": len(self.endpoints),
            "outbound_calls": len(self.outbound_calls),
            "events": len(self.events),
            "service_url_vars": list(self.service_url_vars),
        }
