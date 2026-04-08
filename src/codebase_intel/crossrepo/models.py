"""Cross-repo models — schemas for service dependencies and API contracts.

In a microservice architecture, the most dangerous changes are ones that
break OTHER services. Changing an endpoint in service A silently breaks
service B, C, and D that call it. Nobody knows until production.

This module tracks:
1. Service Registry — what services exist and where they live
2. API Contracts — what endpoints each service exposes
3. Service Dependencies — which services call which endpoints
4. Impact Propagation — "if this endpoint changes, which services break?"

Edge cases:
- Service not yet indexed: discovered incrementally as repos are scanned
- Endpoint renamed: old contract marked deprecated, consumers flagged
- Service removed: all dependents get critical warnings
- Shared types diverge: contract A says field is string, contract B says int
- Async communication (queues): tracked as event contracts, not HTTP
- Multiple versions of same API: tracked with version tags
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ServiceStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    REMOVED = "removed"


class ProtocolType(str, Enum):
    HTTP = "http"
    GRPC = "grpc"
    GRAPHQL = "graphql"
    EVENT = "event"  # Message queue / pub-sub
    INTERNAL = "internal"  # Direct function call (monorepo)


class HttpMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    ANY = "ANY"


class ContractStatus(str, Enum):
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    BREAKING_CHANGE = "breaking_change"
    REMOVED = "removed"


class ServiceInfo(BaseModel):
    """A microservice in the ecosystem."""

    model_config = ConfigDict(frozen=True)

    service_id: str = Field(description="Unique ID: e.g., 'user-module'")
    name: str = Field(description="Human name: 'User Management Service'")
    repo_path: str = Field(description="Absolute path to the repo root")
    status: ServiceStatus = ServiceStatus.ACTIVE
    base_url_env: str = Field(
        default="",
        description="Env var that holds this service's base URL (e.g., USER_SERVICE_URL)",
    )
    tech_stack: str = Field(default="", description="e.g., 'FastAPI + Tortoise ORM'")
    description: str = ""
    indexed_at: datetime | None = None

    @field_validator("indexed_at")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class APIEndpoint(BaseModel):
    """An API endpoint exposed by a service."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Route path: '/api/v1/users/{user_id}'")
    method: HttpMethod = HttpMethod.ANY
    service_id: str = Field(description="Which service exposes this")
    protocol: ProtocolType = ProtocolType.HTTP

    # Schema info (what the endpoint expects/returns)
    request_schema: str = Field(default="", description="Pydantic model name or JSON schema ref")
    response_schema: str = Field(default="", description="Pydantic model name or JSON schema ref")
    description: str = ""

    # Source code location
    file_path: str = ""
    line_number: int | None = None
    function_name: str = ""

    # Contract status
    status: ContractStatus = ContractStatus.ACTIVE
    version: str = Field(default="v1", description="API version")
    deprecated_at: datetime | None = None
    breaking_changes: list[str] = Field(default_factory=list)


class EventContract(BaseModel):
    """An event/message published or consumed by a service."""

    model_config = ConfigDict(frozen=True)

    event_name: str = Field(description="Event type: 'user.created', 'order.completed'")
    service_id: str
    direction: str = Field(description="'publishes' or 'consumes'")
    protocol: ProtocolType = ProtocolType.EVENT
    payload_schema: str = ""
    queue_name: str = ""
    description: str = ""
    file_path: str = ""


class ServiceDependency(BaseModel):
    """A dependency between two services — service A calls service B."""

    model_config = ConfigDict(frozen=True)

    consumer_service: str = Field(description="Service that makes the call")
    provider_service: str = Field(description="Service that handles the call")
    endpoint_path: str = Field(description="Which endpoint is called")
    method: HttpMethod = HttpMethod.ANY

    # How the call is made
    call_pattern: str = Field(
        default="",
        description="How it's called: 'httpx.get(URL)', 'requests.post()', etc.",
    )
    file_path: str = Field(default="", description="Where the call is made")
    line_number: int | None = None

    # Risk assessment
    is_critical: bool = Field(
        default=False,
        description="True if this dependency is in a critical path (auth, payment)",
    )
    has_fallback: bool = Field(
        default=False,
        description="True if there's a fallback/retry mechanism",
    )
    has_circuit_breaker: bool = Field(
        default=False,
        description="True if circuit breaker pattern is implemented",
    )


class CrossRepoImpact(BaseModel):
    """Impact analysis result — what breaks across services when something changes."""

    changed_service: str
    changed_endpoint: str
    changed_file: str = ""
    affected_services: list[ServiceDependency] = Field(default_factory=list)
    risk_level: str = Field(
        default="low",
        description="low / medium / high / critical",
    )
    recommendation: str = ""

    @property
    def affected_count(self) -> int:
        return len(self.affected_services)

    def to_context_string(self) -> str:
        lines = [
            f"## Cross-Service Impact: {self.changed_service}",
            f"Changed: {self.changed_endpoint}",
            f"Risk: **{self.risk_level.upper()}** — {self.affected_count} services affected",
            "",
        ]
        for dep in self.affected_services:
            critical = " [CRITICAL]" if dep.is_critical else ""
            fallback = " (has fallback)" if dep.has_fallback else " (NO fallback)"
            lines.append(
                f"- **{dep.consumer_service}** calls this endpoint{critical}{fallback}"
            )
            if dep.file_path:
                lines.append(f"  at {dep.file_path}:{dep.line_number or '?'}")

        if self.recommendation:
            lines.append(f"\n**Recommendation:** {self.recommendation}")

        return "\n".join(lines)
