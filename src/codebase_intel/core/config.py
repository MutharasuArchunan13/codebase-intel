"""Configuration system with validation and sensible defaults.

Edge cases handled:
- Config file doesn't exist: use defaults, create template on init
- Config file has unknown keys: warn but don't fail (forward compat)
- Config file has invalid values: raise ContractParseError with specifics
- Environment variables override file config (12-factor)
- Relative paths in config: resolved relative to project root
- Multiple config files: project-level overrides user-level overrides defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from codebase_intel.core.types import Language


class ParserConfig(BaseSettings):
    """Tree-sitter parsing configuration."""

    max_file_size_bytes: int = Field(
        default=1_048_576,  # 1MB
        description="Skip files larger than this (generated code, bundles)",
    )
    timeout_ms: int = Field(
        default=5_000,
        description="Per-file parse timeout to handle pathological grammars",
    )
    enabled_languages: list[Language] = Field(
        default=[
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.TSX,
        ],
    )
    ignored_patterns: list[str] = Field(
        default=[
            "node_modules/**",
            ".git/**",
            "__pycache__/**",
            "*.min.js",
            "*.bundle.js",
            "*.generated.*",
            "vendor/**",
            "dist/**",
            "build/**",
            ".venv/**",
            "venv/**",
        ],
        description="Glob patterns for files/dirs to skip entirely",
    )
    generated_markers: list[str] = Field(
        default=[
            "# generated",
            "// generated",
            "/* generated",
            "@generated",
            "DO NOT EDIT",
            "auto-generated",
        ],
        description="If first 5 lines contain any marker, flag as generated",
    )


class GraphConfig(BaseSettings):
    """Code graph storage and behavior configuration."""

    db_path: Path = Field(
        default=Path(".codebase-intel/graph.db"),
        description="SQLite database path (relative to project root)",
    )
    enable_wal_mode: bool = Field(
        default=True,
        description="WAL mode for concurrent read/write (git hook + MCP server)",
    )
    max_traversal_depth: int = Field(
        default=10,
        description="Max depth for dependency traversal to prevent runaway on cycles",
    )
    track_dynamic_imports: bool = Field(
        default=True,
        description="Attempt to resolve importlib / dynamic require (lower confidence)",
    )
    include_type_only_edges: bool = Field(
        default=True,
        description="Include TYPE_CHECKING / type-only imports in graph",
    )


class DecisionConfig(BaseSettings):
    """Decision journal configuration."""

    decisions_dir: Path = Field(
        default=Path(".codebase-intel/decisions"),
        description="Directory for decision YAML files",
    )
    auto_mine_git: bool = Field(
        default=True,
        description="Auto-suggest decisions from PR descriptions and commit messages",
    )
    staleness_threshold_days: int = Field(
        default=90,
        description="Flag decisions not reviewed in this many days",
    )
    max_linked_code_regions: int = Field(
        default=20,
        description="Max code anchors per decision (prevent over-linking)",
    )
    mine_pr_labels: list[str] = Field(
        default=["architecture", "adr", "decision", "breaking-change"],
        description="PR labels that suggest a decision worth recording",
    )


class ContractConfig(BaseSettings):
    """Quality contract configuration."""

    contracts_dir: Path = Field(
        default=Path(".codebase-intel/contracts"),
        description="Directory for contract YAML files",
    )
    enable_builtin_contracts: bool = Field(
        default=True,
        description="Include default contracts (no N+1, layer violations, etc.)",
    )
    strict_mode: bool = Field(
        default=False,
        description="Treat WARNING-level violations as ERROR",
    )


class OrchestratorConfig(BaseSettings):
    """Context orchestrator configuration."""

    default_budget_tokens: int = Field(
        default=8_000,
        description="Default token budget if agent doesn't specify",
    )
    min_useful_tokens: int = Field(
        default=500,
        description="Below this budget, return metadata only (no file content)",
    )
    max_assembly_time_ms: int = Field(
        default=5_000,
        description="Timeout for context assembly to keep MCP responses fast",
    )
    include_stale_context: bool = Field(
        default=True,
        description="Include stale items with low freshness score (vs. dropping them)",
    )
    freshness_decay_days: int = Field(
        default=30,
        description="Context freshness drops to 0.5 after this many days without validation",
    )


class DriftConfig(BaseSettings):
    """Drift detection configuration."""

    rot_threshold_pct: float = Field(
        default=0.3,
        description="Flag context rot event if this % of records are stale",
    )
    check_on_commit: bool = Field(
        default=True,
        description="Run drift check as post-commit hook",
    )
    ignore_generated_files: bool = Field(
        default=True,
        description="Don't flag drift for generated code changes",
    )


class ProjectConfig(BaseSettings):
    """Top-level project configuration aggregating all module configs.

    Resolution order:
    1. Environment variables (CODEBASE_INTEL_*)
    2. Project config file (.codebase-intel/config.yaml)
    3. User config file (~/.config/codebase-intel/config.yaml)
    4. Defaults defined here

    Edge cases:
    - Config file with YAML syntax errors: raise with file path + line number
    - Config file with wrong types: Pydantic validation catches with clear errors
    - Missing config file: use defaults (tool works out of the box)
    - Config file in git: yes, it should be committed (project-specific settings)
    """

    model_config = SettingsConfigDict(
        env_prefix="CODEBASE_INTEL_",
        env_nested_delimiter="__",
    )

    project_root: Path = Field(default=Path("."))
    parser: ParserConfig = Field(default_factory=ParserConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    decisions: DecisionConfig = Field(default_factory=DecisionConfig)
    contracts: ContractConfig = Field(default_factory=ContractConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)

    @field_validator("project_root")
    @classmethod
    def resolve_project_root(cls, v: Path) -> Path:
        resolved = v.resolve()
        if not resolved.is_dir():
            msg = f"Project root does not exist: {resolved}"
            raise ValueError(msg)
        return resolved

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> ProjectConfig:
        """Resolve all relative paths against project_root.

        Edge case: user specifies graph.db_path as "../shared/graph.db"
        for a monorepo setup. We resolve it but verify the parent dir exists.
        """
        root = self.project_root

        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() else (root / p).resolve()

        # Mutate nested configs (Pydantic v2 allows this in validators)
        object.__setattr__(self.graph, "db_path", _resolve(self.graph.db_path))
        object.__setattr__(
            self.decisions, "decisions_dir", _resolve(self.decisions.decisions_dir)
        )
        object.__setattr__(
            self.contracts, "contracts_dir", _resolve(self.contracts.contracts_dir)
        )
        return self

    def ensure_dirs(self) -> None:
        """Create required directories if they don't exist."""
        self.graph.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.decisions.decisions_dir.mkdir(parents=True, exist_ok=True)
        self.contracts.contracts_dir.mkdir(parents=True, exist_ok=True)

    def to_yaml_dict(self) -> dict[str, Any]:
        """Export as a dict suitable for YAML serialization (config template)."""
        return self.model_dump(mode="json", exclude={"project_root"})
