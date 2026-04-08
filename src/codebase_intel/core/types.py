"""Core type definitions shared across all modules.

Design principles:
- Pydantic models for all structured data — never raw dicts
- Immutable by default (frozen=True) — mutation must be explicit
- Content-addressable where possible — xxhash for identity
- Serializable to JSON for MCP transport and SQLite storage

Edge cases addressed in type design:
- File paths: always resolved to absolute, normalized (no symlink ambiguity)
- Timestamps: always UTC, never naive datetimes
- Identifiers: content-hash based where possible (stable across renames)
- Line ranges: 1-indexed to match editor conventions, validated min<=max
- Token counts: approximate — different models tokenize differently
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NodeKind(str, Enum):
    """Types of nodes in the semantic code graph."""

    MODULE = "module"  # A file / compilation unit
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"  # Module-level constants, global state
    INTERFACE = "interface"  # TS interfaces, Python Protocols
    TYPE_ALIAS = "type_alias"
    ENDPOINT = "endpoint"  # HTTP/gRPC/GraphQL endpoints
    CONFIG = "config"  # Config files that affect behavior
    UNKNOWN = "unknown"  # Parsed but unclassifiable


class EdgeKind(str, Enum):
    """Types of relationships between graph nodes."""

    IMPORTS = "imports"  # Static import
    DYNAMIC_IMPORT = "dynamic_import"  # importlib, dynamic require()
    CALLS = "calls"  # Function/method invocation
    INHERITS = "inherits"  # Class inheritance
    IMPLEMENTS = "implements"  # Interface implementation
    INSTANTIATES = "instantiates"  # Object creation
    READS = "reads"  # Reads from variable/config
    WRITES = "writes"  # Mutates variable/state
    DEPENDS_ON = "depends_on"  # Generic dependency (package-level)
    TESTS = "tests"  # Test file → source file relationship
    CONFIGURES = "configures"  # Config file → module it affects
    RE_EXPORTS = "re_exports"  # Barrel file re-exporting


class Language(str, Enum):
    """Supported source languages — 19 languages via tree-sitter-language-pack."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    TSX = "tsx"
    GO = "go"
    RUST = "rust"
    JAVA = "java"
    RUBY = "ruby"
    C = "c"
    CPP = "cpp"
    CSHARP = "c_sharp"
    PHP = "php"
    SWIFT = "swift"
    KOTLIN = "kotlin"
    SCALA = "scala"
    LUA = "lua"
    DART = "dart"
    ELIXIR = "elixir"
    HASKELL = "haskell"
    UNKNOWN = "unknown"  # tracked in graph but not parsed internally


class DecisionStatus(str, Enum):
    """Lifecycle of a decision record."""

    DRAFT = "draft"  # Not yet finalized
    ACTIVE = "active"  # Currently in effect
    SUPERSEDED = "superseded"  # Replaced by another decision
    DEPRECATED = "deprecated"  # Still in code but being removed
    EXPIRED = "expired"  # Past its review date


class ContractSeverity(str, Enum):
    """How strictly a contract rule is enforced."""

    ERROR = "error"  # Must not violate — blocks generation
    WARNING = "warning"  # Should not violate — agent sees it as guidance
    INFO = "info"  # Nice to know — lowest priority context


class DriftLevel(str, Enum):
    """Severity of detected drift."""

    NONE = "none"
    LOW = "low"  # Minor: line numbers shifted
    MEDIUM = "medium"  # Moderate: content changed but structure intact
    HIGH = "high"  # Major: code significantly different
    CRITICAL = "critical"  # Anchor deleted or file removed


# ---------------------------------------------------------------------------
# Value objects (immutable, content-addressed)
# ---------------------------------------------------------------------------


class FileFingerprint(BaseModel):
    """Content-addressable identity for a file at a point in time.

    Edge cases:
    - Renamed files: same content hash → detectable as rename, not delete+create
    - Binary files: we hash them but don't parse internals
    - Empty files: valid fingerprint (empty string hashes consistently)
    - Encoding issues: we read as bytes for hashing, decode for parsing separately
    """

    model_config = ConfigDict(frozen=True)

    path: Path
    content_hash: str = Field(description="xxhash of file bytes")
    size_bytes: int = Field(ge=0)
    last_modified: datetime
    language: Language = Language.UNKNOWN

    @field_validator("last_modified")
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @field_validator("path")
    @classmethod
    def normalize_path(cls, v: Path) -> Path:
        return v.resolve()


class LineRange(BaseModel):
    """A range of lines in a file (1-indexed, inclusive).

    Edge cases:
    - Single-line range: start == end (valid)
    - Full-file range: start=1, end=line_count
    - After refactor: line numbers shift — we re-anchor via content hash
    """

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def start_before_end(self) -> Self:
        if self.start > self.end:
            msg = f"Line range start ({self.start}) must be <= end ({self.end})"
            raise ValueError(msg)
        return self

    @property
    def span(self) -> int:
        return self.end - self.start + 1


class CodeAnchor(BaseModel):
    """Links a non-code artifact (decision, contract) to a specific code location.

    Edge cases:
    - File renamed: content_hash lets us find it at new path
    - Lines shifted: content_hash of the anchored region lets us re-locate
    - File deleted: anchor becomes orphaned — drift detector flags it
    - Function renamed: symbol_name helps re-anchor even if hash changes
    """

    model_config = ConfigDict(frozen=True)

    file_path: Path
    line_range: LineRange | None = None
    symbol_name: str | None = None  # e.g., "RateLimiter.check"
    content_hash: str | None = None  # hash of the anchored code region

    @field_validator("file_path")
    @classmethod
    def normalize_path(cls, v: Path) -> Path:
        return v.resolve()

    def is_orphaned(self, existing_paths: set[Path]) -> bool:
        """Check if the anchor's file still exists."""
        return self.file_path.resolve() not in existing_paths


class TokenBudget(BaseModel):
    """Token budget for context assembly.

    Edge cases:
    - Zero budget: return empty context with metadata only
    - Budget smaller than minimum useful context: return highest-priority items
      with a truncation warning
    - Model-specific tokenization: budget is approximate since different models
      tokenize differently. We use tiktoken cl100k_base as a reasonable baseline
      and include a safety margin.
    """

    model_config = ConfigDict(frozen=True)

    total: int = Field(gt=0, description="Total tokens available for context")
    reserved_for_response: int = Field(
        default=0,
        ge=0,
        description="Tokens to reserve for the agent's response",
    )
    safety_margin_pct: float = Field(
        default=0.1,
        ge=0,
        le=0.5,
        description="Percentage to hold back for tokenization variance",
    )

    @property
    def usable(self) -> int:
        """Actual tokens available for context after reserves."""
        raw = self.total - self.reserved_for_response
        margin = int(raw * self.safety_margin_pct)
        return max(0, raw - margin)


# ---------------------------------------------------------------------------
# Graph node & edge
# ---------------------------------------------------------------------------


class GraphNode(BaseModel):
    """A node in the semantic code graph.

    Edge cases:
    - Generated code: marked with is_generated=True, lower trust weight
    - Vendored/third-party: marked with is_external=True, read-only context
    - Test files: marked with is_test=True, linked to source via TESTS edge
    - Barrel/index files: may have many RE_EXPORTS edges, low own content
    """

    model_config = ConfigDict(frozen=True)

    node_id: str = Field(description="Stable ID: hash of (path, kind, name)")
    kind: NodeKind
    name: str
    qualified_name: str = Field(
        description="Full path: module.Class.method"
    )
    file_path: Path
    line_range: LineRange | None = None
    language: Language = Language.UNKNOWN
    content_hash: str | None = None
    docstring: str | None = None

    is_generated: bool = False
    is_external: bool = False
    is_test: bool = False
    is_entry_point: bool = False

    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def make_id(file_path: Path, kind: NodeKind, name: str) -> str:
        """Deterministic node ID from its identity components."""
        raw = f"{file_path.resolve()}:{kind.value}:{name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class GraphEdge(BaseModel):
    """A directed edge between two graph nodes.

    Edge cases:
    - Conditional edges: import inside if TYPE_CHECKING — marked is_type_only
    - Dynamic edges: importlib.import_module() — marked with confidence < 1.0
    - Circular edges: A → B → A — valid, traversal must handle
    - Cross-language edges: Python calling C extension — low confidence
    """

    model_config = ConfigDict(frozen=True)

    source_id: str
    target_id: str
    kind: EdgeKind
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 = static import, <1.0 = inferred/dynamic",
    )
    is_type_only: bool = Field(
        default=False,
        description="True for TYPE_CHECKING imports, type annotations only",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Context assembly types
# ---------------------------------------------------------------------------


class ContextPriority(str, Enum):
    """Priority levels for context items during budget allocation."""

    CRITICAL = "critical"  # Must include — directly referenced files
    HIGH = "high"  # Should include — immediate dependencies, active decisions
    MEDIUM = "medium"  # Include if budget allows — transitive deps, older decisions
    LOW = "low"  # Nice to have — tangential context, info-level contracts


class ContextItem(BaseModel):
    """A single item in the assembled context payload.

    This is the universal wrapper — every piece of context (file content,
    decision record, contract rule) gets wrapped in this for the orchestrator
    to prioritize and budget.
    """

    source: str = Field(description="Which module provided this: graph|decisions|contracts")
    item_type: str = Field(description="Specific type: file_content|decision|contract_rule|warning")
    priority: ContextPriority
    estimated_tokens: int = Field(ge=0)
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    freshness_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="1.0 = just validated, 0.0 = very stale",
    )


class AssembledContext(BaseModel):
    """The final context payload sent to an AI agent.

    Edge cases:
    - Empty context: valid if the task has no relevant files (new greenfield code)
    - Truncated context: items were dropped due to budget — truncated=True with
      a summary of what was dropped
    - Conflicting context: contains contradictions — conflicts list populated
    - Partial context: some modules unavailable — warnings list populated
    """

    items: list[ContextItem] = Field(default_factory=list)
    total_tokens: int = 0
    budget_tokens: int = 0
    truncated: bool = False
    dropped_count: int = Field(
        default=0,
        description="Number of items dropped due to budget",
    )
    conflicts: list[str] = Field(
        default_factory=list,
        description="Human-readable conflict descriptions",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Degradation notices (partial init, stale data, etc.)",
    )
    assembly_time_ms: float = 0.0
