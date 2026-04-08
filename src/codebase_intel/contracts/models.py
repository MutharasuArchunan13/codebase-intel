"""Quality contract models — the schema for expressing "what good looks like."

A quality contract defines rules that code should follow. Unlike linters
(which check syntax/style), contracts enforce architectural patterns,
project-specific conventions, and anti-pattern avoidance.

Three layers:
1. Architectural constraints: structural rules (layer violations, dependency direction)
2. Pattern library: approved and forbidden code patterns
3. Quality gates: measurable thresholds (complexity, coverage, function length)

Edge cases in design:
- Conflicting contracts: Contract A says "max 50 lines per function" but Contract B
  says "no helper functions for one-time logic." For a 60-line function, these conflict.
  Resolution: priority system (P0 > P1 > P2). Same priority = flag as conflict.
- Scope exclusions: "all code except tests", "only src/api/**"
- Gradual migration: old pattern and new pattern coexist during transition.
  Contracts support a `migration_deadline` after which the old pattern is an error.
- Framework-specific: a "React patterns" contract pack shouldn't fire on Python files.
- Runtime-dependent rules: "no N+1 queries" — can only partially check statically
  (detect loop+query pattern, but may false-positive on batched queries).
- AI-specific anti-patterns: rules targeting common AI generation mistakes.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codebase_intel.core.types import ContractSeverity, Language


class RuleKind(str, Enum):
    """Types of quality rules."""

    ARCHITECTURAL = "architectural"  # Structural: layer violations, dependency direction
    PATTERN = "pattern"  # Code pattern: approved vs forbidden constructs
    THRESHOLD = "threshold"  # Measurable: complexity, length, coverage
    AI_ANTIPATTERN = "ai_antipattern"  # AI-specific: hallucinated imports, over-abstraction


class ScopeFilter(BaseModel):
    """Defines which files/modules a contract applies to.

    Edge cases:
    - Empty include list: contract applies to everything (minus excludes)
    - Only exclude list: applies to everything except the excluded paths
    - Both include and exclude: include first, then subtract excludes
    - Language filter: only applies to files of specific languages
    - Directory depth: "src/api/**" vs "src/api/*" (recursive vs single level)
    """

    model_config = ConfigDict(frozen=True)

    include_patterns: list[str] = Field(
        default_factory=list,
        description="Glob patterns for files this contract applies to. Empty = all files.",
    )
    exclude_patterns: list[str] = Field(
        default_factory=lambda: [
            "node_modules/**",
            "*.min.js",
            "*.generated.*",
            "vendor/**",
            "dist/**",
        ],
        description="Glob patterns for files to exclude from this contract.",
    )
    languages: list[Language] = Field(
        default_factory=list,
        description="Restrict to specific languages. Empty = all languages.",
    )
    exclude_tests: bool = Field(
        default=False,
        description="Exclude test files from this contract.",
    )
    exclude_generated: bool = Field(
        default=True,
        description="Exclude generated files from this contract.",
    )

    def matches(
        self,
        file_path: Path,
        language: Language = Language.UNKNOWN,
        is_test: bool = False,
        is_generated: bool = False,
    ) -> bool:
        """Check if a file falls within this contract's scope."""
        import fnmatch

        if self.exclude_tests and is_test:
            return False
        if self.exclude_generated and is_generated:
            return False
        if self.languages and language not in self.languages:
            return False

        path_str = str(file_path)

        # Check excludes first
        if any(fnmatch.fnmatch(path_str, p) for p in self.exclude_patterns):
            return False

        # If no include patterns, match everything not excluded
        if not self.include_patterns:
            return True

        return any(fnmatch.fnmatch(path_str, p) for p in self.include_patterns)


class PatternExample(BaseModel):
    """A code example showing the approved or forbidden pattern.

    Providing concrete examples is critical — agents need to see
    "do this instead" not just "don't do that."
    """

    model_config = ConfigDict(frozen=True)

    code: str
    language: Language = Language.UNKNOWN
    description: str = ""
    is_approved: bool = True  # True = do this, False = don't do this


class ContractRule(BaseModel):
    """A single rule within a quality contract.

    Edge cases:
    - Rule with regex pattern: must compile without error at load time
    - Rule with threshold: value must be positive and reasonable
    - Rule with examples: examples should match the rule's language filter
    - Rule in migration mode: has deadline, old pattern is WARNING before
      deadline and ERROR after
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Rule ID within this contract, e.g., 'no-direct-db'")
    name: str = Field(description="Human-readable rule name")
    description: str = Field(description="What this rule checks and why")
    kind: RuleKind
    severity: ContractSeverity = ContractSeverity.WARNING

    # Detection configuration
    pattern: str | None = Field(
        default=None,
        description="Regex pattern to detect violations (for PATTERN rules)",
    )
    anti_pattern: str | None = Field(
        default=None,
        description="Regex pattern that SHOULD be present (absence = violation)",
    )
    threshold_metric: str | None = Field(
        default=None,
        description="Metric name for THRESHOLD rules: 'max_lines', 'max_complexity', etc.",
    )
    threshold_value: float | None = Field(
        default=None,
        description="Maximum allowed value for the threshold metric",
    )

    # Context for agents
    examples: list[PatternExample] = Field(
        default_factory=list,
        description="Approved and forbidden code examples",
    )
    fix_suggestion: str | None = Field(
        default=None,
        description="How to fix a violation (shown to agents)",
    )

    # Migration support
    migration_deadline: datetime | None = Field(
        default=None,
        description="After this date, old pattern becomes an ERROR (gradual migration)",
    )
    replaces_pattern: str | None = Field(
        default=None,
        description="The old pattern being migrated away from",
    )

    @field_validator("migration_deadline")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @property
    def effective_severity(self) -> ContractSeverity:
        """Severity adjusted for migration deadlines.

        Before deadline: original severity (usually WARNING)
        After deadline: ERROR (migration period is over)
        """
        if self.migration_deadline and datetime.now(UTC) > self.migration_deadline:
            return ContractSeverity.ERROR
        return self.severity


class QualityContract(BaseModel):
    """A named collection of quality rules.

    Contracts are the top-level organizational unit. A project might have:
    - "core-architecture" contract (layer rules)
    - "react-patterns" contract (frontend-specific)
    - "api-conventions" contract (endpoint standards)
    - "ai-guardrails" contract (AI-specific anti-patterns)

    Each contract has a scope (which files it applies to) and a priority
    (for conflict resolution).
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Contract ID, e.g., 'core-architecture'")
    name: str = Field(description="Human-readable name")
    description: str = Field(description="What this contract enforces and why")
    version: str = Field(default="1.0.0")
    priority: int = Field(
        default=100,
        ge=0,
        le=1000,
        description="Higher priority wins in conflicts. 0=lowest, 1000=highest.",
    )

    scope: ScopeFilter = Field(default_factory=ScopeFilter)
    rules: list[ContractRule] = Field(default_factory=list)

    # Metadata
    author: str = Field(default="unknown")
    tags: list[str] = Field(default_factory=list)
    is_builtin: bool = Field(
        default=False,
        description="True for contracts shipped with codebase-intel",
    )

    def rules_for_file(
        self,
        file_path: Path,
        language: Language = Language.UNKNOWN,
        is_test: bool = False,
        is_generated: bool = False,
    ) -> list[ContractRule]:
        """Get applicable rules for a specific file.

        Edge case: contract scope matches but individual rules may
        have language-specific patterns. A Python regex won't match
        in a TypeScript file. We return all rules and let the evaluator
        handle language-specific matching.
        """
        if not self.scope.matches(file_path, language, is_test, is_generated):
            return []
        return list(self.rules)

    def to_context_string(self, verbose: bool = False) -> str:
        """Serialize for inclusion in agent context.

        Compact mode: rule names and descriptions only.
        Verbose mode: includes examples and fix suggestions.
        """
        lines = [
            f"## Contract: {self.name} [{self.id}]",
            f"Priority: {self.priority} | Rules: {len(self.rules)}",
            f"{self.description}",
            "",
        ]

        for rule in self.rules:
            severity_badge = {
                ContractSeverity.ERROR: "ERROR",
                ContractSeverity.WARNING: "WARN",
                ContractSeverity.INFO: "INFO",
            }[rule.effective_severity]

            lines.append(f"- [{severity_badge}] **{rule.name}**: {rule.description}")

            if verbose and rule.fix_suggestion:
                lines.append(f"  Fix: {rule.fix_suggestion}")

            if verbose and rule.examples:
                for ex in rule.examples:
                    label = "DO" if ex.is_approved else "DON'T"
                    lines.append(f"  {label}: {ex.description}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in contract definitions (shipped with the tool)
# ---------------------------------------------------------------------------


def builtin_ai_guardrails() -> QualityContract:
    """Contract for detecting common AI code generation mistakes.

    These are patterns that AI agents frequently produce but humans
    wouldn't write. Detecting them before they reach the codebase
    is a key value proposition.
    """
    return QualityContract(
        id="ai-guardrails",
        name="AI Code Generation Guardrails",
        description=(
            "Detects common anti-patterns in AI-generated code: "
            "hallucinated imports, over-abstraction, unnecessary error handling, "
            "verbose comments restating code, and speculative features."
        ),
        priority=500,
        is_builtin=True,
        tags=["ai", "quality", "guardrails"],
        rules=[
            ContractRule(
                id="no-hallucinated-imports",
                name="No hallucinated imports",
                description=(
                    "AI agents sometimes import modules that don't exist in the project. "
                    "Verify all imports resolve to actual files or installed packages."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.ERROR,
                fix_suggestion="Check if the imported module exists. Use the code graph to verify.",
            ),
            ContractRule(
                id="no-over-abstraction",
                name="No premature abstraction",
                description=(
                    "AI tends to create unnecessary base classes, factory patterns, "
                    "and utility wrappers for one-time operations. Only abstract when "
                    "there are 3+ concrete uses."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.WARNING,
                fix_suggestion="Inline the logic. Create abstractions only when there are 3+ users.",
            ),
            ContractRule(
                id="no-unnecessary-error-handling",
                name="No redundant error handling",
                description=(
                    "AI often wraps code in try/except or if-null checks for conditions "
                    "that can't occur (e.g., checking if a required field is None after "
                    "Pydantic validation). Trust the type system and framework guarantees."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.WARNING,
                fix_suggestion="Remove error handling for impossible conditions. Trust types and validation.",
            ),
            ContractRule(
                id="no-restating-comments",
                name="No comments that restate code",
                description=(
                    "AI generates excessive comments like '# Increment counter' above 'counter += 1'. "
                    "Comments should explain WHY, never WHAT."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.INFO,
                pattern=r"#\s*(set|get|create|update|delete|increment|initialize|return)\s",
                fix_suggestion="Remove comments that describe what the code does. Only keep WHY comments.",
            ),
            ContractRule(
                id="no-speculative-features",
                name="No unrequested features",
                description=(
                    "AI often adds extra configuration, feature flags, or extensibility "
                    "points that weren't asked for. YAGNI — You Ain't Gonna Need It."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.WARNING,
                fix_suggestion="Remove features not explicitly requested. Build exactly what was asked.",
            ),
            ContractRule(
                id="no-excessive-logging",
                name="No excessive logging",
                description=(
                    "AI often adds a log statement for every operation. Only log at "
                    "meaningful boundaries: errors, external calls, state transitions."
                ),
                kind=RuleKind.AI_ANTIPATTERN,
                severity=ContractSeverity.INFO,
                fix_suggestion="Keep logging at meaningful boundaries only. Remove noise logs.",
            ),
        ],
    )


def builtin_architecture_rules() -> QualityContract:
    """Generic architectural rules applicable to most projects."""
    return QualityContract(
        id="architecture-basics",
        name="Basic Architecture Rules",
        description="Fundamental architectural constraints: layer separation, dependency direction, no circular imports.",
        priority=400,
        is_builtin=True,
        tags=["architecture"],
        rules=[
            ContractRule(
                id="no-circular-imports",
                name="No circular import chains",
                description="Circular imports cause initialization issues and indicate tangled architecture.",
                kind=RuleKind.ARCHITECTURAL,
                severity=ContractSeverity.WARNING,
                fix_suggestion="Break the cycle by extracting shared types into a separate module.",
            ),
            ContractRule(
                id="no-god-files",
                name="No files exceeding 500 lines",
                description="Files over 500 lines are hard to navigate and usually need splitting.",
                kind=RuleKind.THRESHOLD,
                severity=ContractSeverity.WARNING,
                threshold_metric="max_lines",
                threshold_value=500,
                fix_suggestion="Split into focused modules by responsibility.",
            ),
            ContractRule(
                id="no-god-functions",
                name="No functions exceeding 50 lines",
                description="Long functions are hard to test and understand. Extract sub-routines.",
                kind=RuleKind.THRESHOLD,
                severity=ContractSeverity.WARNING,
                threshold_metric="max_function_lines",
                threshold_value=50,
                fix_suggestion="Extract helper functions or use early returns to reduce length.",
            ),
        ],
    )
