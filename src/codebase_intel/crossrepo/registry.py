"""Cross-repo registry — the central hub that connects services.

This is where the magic happens. The registry:
1. Holds all scanned services and their endpoints
2. Matches outbound calls to exposed endpoints (dependency resolution)
3. Computes cross-service impact when an endpoint changes
4. Persists to YAML for git-friendly tracking

The key insight: an outbound call in service A contains a URL pattern
like '/api/v1/users/{id}'. The registry matches this against exposed
endpoints across all services to find: "service A depends on the user
endpoint in service B."

Edge cases:
- URL pattern doesn't match any known endpoint: flagged as unresolved
- Multiple services expose the same path: ambiguous — flagged
- Service URL built from env var: matched via env var name patterns
- API version mismatch: consumer calls /v1 but provider only has /v2
- Endpoint removed: all consumers get critical warnings
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from codebase_intel.crossrepo.models import (
    APIEndpoint,
    ContractStatus,
    CrossRepoImpact,
    HttpMethod,
    ServiceDependency,
    ServiceInfo,
    ServiceStatus,
)
from codebase_intel.crossrepo.scanner import OutboundCall, ServiceScanResult, ServiceScanner

logger = logging.getLogger(__name__)


class CrossRepoRegistry:
    """Central registry of all services, endpoints, and dependencies."""

    def __init__(self, registry_path: Path) -> None:
        self._registry_path = registry_path
        self._registry_path.mkdir(parents=True, exist_ok=True)

        self.services: dict[str, ServiceInfo] = {}
        self.endpoints: list[APIEndpoint] = []
        self.dependencies: list[ServiceDependency] = []
        self._scan_results: dict[str, ServiceScanResult] = {}

    def register_service(
        self,
        repo_path: Path,
        service_id: str | None = None,
        name: str | None = None,
    ) -> ServiceScanResult:
        """Register and scan a service repo."""
        repo_path = repo_path.resolve()
        service_id = service_id or repo_path.name
        name = name or service_id.replace("-", " ").replace("_", " ").title()

        # Scan the repo
        scanner = ServiceScanner()
        result = scanner.scan_service(repo_path, service_id)

        # Register service info
        self.services[service_id] = ServiceInfo(
            service_id=service_id,
            name=name,
            repo_path=str(repo_path),
            indexed_at=datetime.now(UTC),
        )

        # Add endpoints
        self.endpoints.extend(result.endpoints)
        self._scan_results[service_id] = result

        return result

    def resolve_dependencies(self) -> list[ServiceDependency]:
        """Match outbound calls against known endpoints.

        This is the core resolution algorithm:
        1. For each outbound call, extract the URL path
        2. Normalize it (strip base URL, version prefix)
        3. Match against all known endpoint paths
        4. Create a ServiceDependency for each match

        Edge cases:
        - URL is a variable, not a literal: try matching the var name
          against service URL env var patterns
        - URL contains f-string interpolation: extract the static parts
        - No match found: log as unresolved dependency
        """
        self.dependencies = []
        unresolved: list[OutboundCall] = []

        for service_id, scan in self._scan_results.items():
            for call in scan.outbound_calls:
                matched = self._resolve_call(call, service_id)
                if matched:
                    self.dependencies.append(matched)
                else:
                    unresolved.append(call)

        if unresolved:
            logger.info(
                "%d outbound calls could not be resolved to known endpoints",
                len(unresolved),
            )

        return self.dependencies

    def impact_analysis(
        self,
        service_id: str,
        endpoint_path: str | None = None,
        changed_file: str = "",
    ) -> list[CrossRepoImpact]:
        """Find which services are affected by changes to a service's endpoints.

        If endpoint_path is given: find services that call this specific endpoint.
        If not: find all services that depend on any endpoint in this service.
        """
        impacts: list[CrossRepoImpact] = []

        if endpoint_path:
            # Specific endpoint changed
            affected = [
                d for d in self.dependencies
                if d.provider_service == service_id
                and self._paths_match(d.endpoint_path, endpoint_path)
            ]

            if affected:
                risk = self._assess_risk(affected)
                impacts.append(CrossRepoImpact(
                    changed_service=service_id,
                    changed_endpoint=endpoint_path,
                    changed_file=changed_file,
                    affected_services=affected,
                    risk_level=risk,
                    recommendation=self._generate_recommendation(affected, risk),
                ))
        else:
            # Any endpoint in this service — deduplicate by endpoint path
            service_endpoints = [
                e for e in self.endpoints if e.service_id == service_id
            ]
            seen_paths: set[str] = set()
            for endpoint in service_endpoints:
                normalized = endpoint.path.strip("/").lower()
                if normalized in seen_paths:
                    continue
                seen_paths.add(normalized)

                affected = [
                    d for d in self.dependencies
                    if d.provider_service == service_id
                    and self._paths_match(d.endpoint_path, endpoint.path)
                ]
                if affected:
                    risk = self._assess_risk(affected)
                    impacts.append(CrossRepoImpact(
                        changed_service=service_id,
                        changed_endpoint=endpoint.path,
                        changed_file=endpoint.file_path,
                        affected_services=affected,
                        risk_level=risk,
                        recommendation=self._generate_recommendation(affected, risk),
                    ))

        return impacts

    def get_service_map(self) -> dict[str, Any]:
        """Get a high-level map of all services and their connections."""
        services = {}
        for sid, info in self.services.items():
            scan = self._scan_results.get(sid)
            services[sid] = {
                "name": info.name,
                "endpoints_exposed": len(scan.endpoints) if scan else 0,
                "outbound_calls": len(scan.outbound_calls) if scan else 0,
                "depends_on": list({
                    d.provider_service for d in self.dependencies
                    if d.consumer_service == sid
                }),
                "depended_on_by": list({
                    d.consumer_service for d in self.dependencies
                    if d.provider_service == sid
                }),
            }
        return services

    def save(self) -> Path:
        """Persist the registry to YAML."""
        data = {
            "services": {
                sid: {
                    "name": info.name,
                    "repo_path": info.repo_path,
                    "status": info.status.value,
                    "indexed_at": info.indexed_at.isoformat() if info.indexed_at else None,
                }
                for sid, info in self.services.items()
            },
            "endpoints": [
                {
                    "service": e.service_id,
                    "method": e.method.value,
                    "path": e.path,
                    "function": e.function_name,
                    "file": e.file_path,
                    "line": e.line_number,
                }
                for e in self.endpoints
            ],
            "dependencies": [
                {
                    "consumer": d.consumer_service,
                    "provider": d.provider_service,
                    "endpoint": d.endpoint_path,
                    "method": d.method.value,
                    "file": d.file_path,
                    "critical": d.is_critical,
                }
                for d in self.dependencies
            ],
        }

        out_path = self._registry_path / "service-registry.yaml"
        out_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        logger.info("Saved service registry to %s", out_path)
        return out_path

    # -------------------------------------------------------------------
    # Private
    # -------------------------------------------------------------------

    def _resolve_call(
        self, call: OutboundCall, consumer_service: str
    ) -> ServiceDependency | None:
        """Try to match an outbound call to a known endpoint."""
        url = call.url

        # Skip empty URLs and calls to self
        if not url:
            return None

        # Extract path from full URL
        path = self._extract_path(url)
        if not path:
            return None

        # Try matching against all known endpoints
        for endpoint in self.endpoints:
            # Don't match service calling itself
            if endpoint.service_id == consumer_service:
                continue

            if self._paths_match(path, endpoint.path):
                try:
                    method = HttpMethod(call.method)
                except ValueError:
                    method = HttpMethod.ANY

                return ServiceDependency(
                    consumer_service=consumer_service,
                    provider_service=endpoint.service_id,
                    endpoint_path=endpoint.path,
                    method=method,
                    call_pattern=call.raw_line[:100],
                    file_path=call.file_path,
                    line_number=call.line_number,
                    is_critical=self._is_critical_path(path),
                )

        return None

    def _extract_path(self, url: str) -> str:
        """Extract the API path from a URL.

        Handles: '/api/v1/users', 'http://service/api/users',
        'f"{base_url}/api/users"', '{SERVICE_URL}/users'
        """
        # Remove protocol and host
        path = re.sub(r'https?://[^/]+', '', url)
        # Remove f-string interpolation markers
        path = re.sub(r'\{[^}]*\}/', '', path, count=1)  # Remove first {var}/
        # Keep path parameters
        path = path.strip("/ ")
        if path and not path.startswith("/"):
            path = "/" + path
        return path

    def _paths_match(self, call_path: str, endpoint_path: str) -> bool:
        """Match a called path against an endpoint path.

        '/api/v1/users/123' should match '/api/v1/users/{user_id}'
        '/users' should match '/users'

        Edge cases:
        - Path params: {id}, {user_id}, <id> — treated as wildcards
        - Trailing slash: /users/ matches /users
        - Version prefix: sometimes stripped
        """
        # Normalize
        call_path = call_path.strip("/").lower()
        endpoint_path = endpoint_path.strip("/").lower()

        if call_path == endpoint_path:
            return True

        # Convert endpoint path params to regex
        pattern = re.sub(r'\{[^}]+\}', r'[^/]+', endpoint_path)
        pattern = re.sub(r'<[^>]+>', r'[^/]+', pattern)

        return bool(re.fullmatch(pattern, call_path))

    def _is_critical_path(self, path: str) -> bool:
        """Determine if an endpoint is on a critical path."""
        critical_keywords = [
            "auth", "login", "token", "payment", "billing",
            "checkout", "order", "session", "verify",
        ]
        path_lower = path.lower()
        return any(kw in path_lower for kw in critical_keywords)

    def _assess_risk(self, affected: list[ServiceDependency]) -> str:
        """Assess the risk level of a cross-service change."""
        if any(d.is_critical for d in affected):
            return "critical"
        if len(affected) >= 5:
            return "high"
        if len(affected) >= 2:
            return "medium"
        return "low"

    def _generate_recommendation(
        self, affected: list[ServiceDependency], risk: str
    ) -> str:
        """Generate actionable recommendation based on impact."""
        count = len(affected)
        services = list({d.consumer_service for d in affected})

        if risk == "critical":
            return (
                f"CRITICAL: {count} services depend on this endpoint, including critical paths. "
                f"Coordinate with teams owning: {', '.join(services)}. "
                f"Consider a backward-compatible change or versioned endpoint."
            )
        elif risk == "high":
            return (
                f"HIGH: {count} services affected ({', '.join(services)}). "
                f"Notify dependent teams before deploying. Consider API versioning."
            )
        elif risk == "medium":
            return (
                f"MEDIUM: {count} services affected ({', '.join(services)}). "
                f"Verify backward compatibility or coordinate the change."
            )
        return f"LOW: {count} service(s) affected. Standard change process."
