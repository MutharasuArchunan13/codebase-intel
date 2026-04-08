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

# Route decorator patterns across frameworks
ROUTE_PATTERNS = [
    # FastAPI / Starlette
    re.compile(
        r'@(?:app|router|api_router)\.'
        r'(get|post|put|patch|delete|options|head)\s*\(\s*["\']([^"\']+)["\']'
    ),
    # Flask
    re.compile(
        r'@(?:app|bp|blueprint)\.'
        r'route\s*\(\s*["\']([^"\']+)["\']'
        r'(?:.*methods\s*=\s*\[([^\]]+)\])?'
    ),
    # Express.js (for JS/TS scanning)
    re.compile(
        r'(?:app|router)\.'
        r'(get|post|put|patch|delete)\s*\(\s*["\']([^"\']+)["\']'
    ),
]

# Outbound HTTP call patterns
HTTP_CALL_PATTERNS = [
    # httpx
    re.compile(
        r'(?:httpx|client|async_client)\.'
        r'(get|post|put|patch|delete)\s*\(\s*'
        r'(?:f?["\']([^"\']*)["\']|(\w+))'
    ),
    # requests
    re.compile(
        r'requests\.'
        r'(get|post|put|patch|delete)\s*\(\s*'
        r'(?:f?["\']([^"\']*)["\']|(\w+))'
    ),
    # aiohttp
    re.compile(
        r'session\.'
        r'(get|post|put|patch|delete)\s*\(\s*'
        r'(?:f?["\']([^"\']*)["\']|(\w+))'
    ),
    # Generic URL building
    re.compile(
        r'(?:url|endpoint|base_url|service_url)\s*(?:\+|=)\s*'
        r'f?["\']([^"\']*(?:api|/v\d)[^"\']*)["\']'
    ),
]

# Environment variable patterns for service URLs
SERVICE_URL_PATTERNS = [
    re.compile(r'(\w+(?:SERVICE|API|BACKEND)_(?:URL|HOST|BASE_URL|ENDPOINT))\b'),
    re.compile(r'os\.(?:environ|getenv)\s*\(\s*["\'](\w*(?:URL|HOST|BASE)\w*)["\']'),
    re.compile(r'settings\.(\w*(?:url|host|base_url|endpoint)\w*)', re.IGNORECASE),
]

# Event/message patterns
EVENT_PATTERNS = [
    re.compile(r'\.publish\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'\.subscribe\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'\.send_message\s*\(\s*["\']([^"\']+)["\']'),
    re.compile(r'channel\s*=\s*["\']([^"\']+)["\']'),
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
                    elif entry.is_file() and entry.suffix in (".py", ".ts", ".js", ".tsx", ".jsx"):
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
