"""Exception hierarchy for codebase-intel.

Design principles:
- Every exception carries structured context (not just a message string)
- Exceptions map to specific failure modes, never generic catch-alls
- Each exception knows whether it's retryable and what recovery action to suggest
- Exceptions are grouped by module to avoid collision and aid filtering

Edge cases handled:
- Partial initialization (graph exists but decisions don't)
- Concurrent access conflicts (two processes updating the same graph)
- Corrupt storage (SQLite file damaged, YAML malformed)
- Resource exhaustion (token budget exceeded, file too large)
- External dependency failures (tree-sitter crash, git unavailable)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Severity(Enum):
    """How urgently this error needs attention."""

    WARNING = "warning"  # Degraded but functional — log and continue
    ERROR = "error"  # Operation failed — caller must handle
    FATAL = "fatal"  # System unusable — abort and report


class RecoveryHint(Enum):
    """Machine-readable recovery suggestions for agents and CLI."""

    RETRY = "retry"  # Transient failure, try again
    REINITIALIZE = "reinitialize"  # Run `codebase-intel init` again
    REDUCE_SCOPE = "reduce_scope"  # Narrow the query or task
    MANUAL_FIX = "manual_fix"  # Human intervention needed
    SKIP = "skip"  # Safe to skip this item and continue
    UPDATE_CONFIG = "update_config"  # Configuration needs adjustment


@dataclass
class ErrorContext:
    """Structured context attached to every exception.

    Agents can parse this to make intelligent recovery decisions
    rather than pattern-matching on error message strings.
    """

    file_path: Path | None = None
    line_range: tuple[int, int] | None = None
    component: str = ""  # Which module raised this
    operation: str = ""  # What was being attempted
    details: dict[str, Any] = field(default_factory=dict)


class CodebaseIntelError(Exception):
    """Base exception — all module exceptions inherit from this.

    Every exception in the system carries:
    - severity: how bad is it
    - recovery: what should the caller do
    - context: structured data about what went wrong
    """

    severity: Severity = Severity.ERROR
    recovery: RecoveryHint = RecoveryHint.MANUAL_FIX

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        super().__init__(message)
        self.context = context or ErrorContext()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for MCP/JSON responses."""
        return {
            "error": type(self).__name__,
            "message": str(self),
            "severity": self.severity.value,
            "recovery": self.recovery.value,
            "context": {
                "file_path": str(self.context.file_path) if self.context.file_path else None,
                "line_range": self.context.line_range,
                "component": self.context.component,
                "operation": self.context.operation,
                "details": self.context.details,
            },
        }


# ---------------------------------------------------------------------------
# Storage errors
# ---------------------------------------------------------------------------


class StorageError(CodebaseIntelError):
    """Base for all storage-layer failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "storage"
        super().__init__(message, ctx)


class StorageCorruptError(StorageError):
    """SQLite file or YAML document is unreadable.

    Edge case: partially written file from a crash mid-write.
    Recovery: re-initialize from scratch (data can be rebuilt from git).
    """

    severity = Severity.ERROR
    recovery = RecoveryHint.REINITIALIZE


class StorageConcurrencyError(StorageError):
    """Two processes tried to write simultaneously.

    Edge case: git hook and MCP server both updating the graph.
    SQLite handles this via WAL mode, but we still need to detect
    and retry at the application level.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.RETRY


class StorageMigrationError(StorageError):
    """Database schema version mismatch.

    Edge case: user updated codebase-intel but hasn't migrated
    their existing database. We must not silently drop data.
    """

    severity = Severity.ERROR
    recovery = RecoveryHint.REINITIALIZE


# ---------------------------------------------------------------------------
# Graph errors
# ---------------------------------------------------------------------------


class GraphError(CodebaseIntelError):
    """Base for code graph failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "graph"
        super().__init__(message, ctx)


class ParseError(GraphError):
    """Tree-sitter failed to parse a file.

    Edge cases:
    - Binary file misidentified as source code
    - File with mixed encodings (UTF-8 + Latin-1 in same file)
    - Syntax errors in user's code (must still extract partial info)
    - Generated code with unusual constructs (protobuf, codegen)
    - Files exceeding the size threshold (e.g., 10MB generated file)

    Recovery: skip this file, log warning, continue with partial graph.
    The graph must be useful even if some files can't be parsed.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.SKIP


class CircularDependencyError(GraphError):
    """Detected a circular dependency chain.

    Edge case: A → B → C → A. This is legitimate in many codebases
    (Python allows it with deferred imports). We must:
    1. Record the cycle in the graph (it's real information)
    2. Not infinite-loop during traversal
    3. Flag it as a quality concern without blocking analysis

    This is a WARNING, not an ERROR — cycles are common and valid.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.SKIP

    def __init__(
        self, cycle_path: list[str], context: ErrorContext | None = None
    ) -> None:
        self.cycle_path = cycle_path
        message = f"Circular dependency: {' → '.join(cycle_path)}"
        super().__init__(message, context)


class UnsupportedLanguageError(GraphError):
    """No tree-sitter grammar available for this file type.

    Edge case: user has .proto, .graphql, .sql files — we can't parse them
    but we should still track them as nodes in the graph (without internal
    structure). The graph should degrade gracefully, not fail entirely.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.SKIP


# ---------------------------------------------------------------------------
# Decision errors
# ---------------------------------------------------------------------------


class DecisionError(CodebaseIntelError):
    """Base for decision journal failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "decisions"
        super().__init__(message, ctx)


class DecisionConflictError(DecisionError):
    """Two active decisions contradict each other.

    Edge case: DEC-042 says "use token bucket" and DEC-058 says
    "use sliding window" for the same module. This happens when:
    - Different team members made decisions at different times
    - A decision was superseded but not marked as such
    - Cross-repo decisions conflict

    Recovery: surface both to the agent with conflict metadata
    so it can ask the developer which one to follow.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.MANUAL_FIX

    def __init__(
        self,
        decision_a: str,
        decision_b: str,
        context: ErrorContext | None = None,
    ) -> None:
        self.decision_a = decision_a
        self.decision_b = decision_b
        message = f"Conflicting decisions: {decision_a} vs {decision_b}"
        super().__init__(message, context)


class StaleDecisionError(DecisionError):
    """Decision references code that has changed significantly.

    Edge case: decision links to src/auth/middleware.py:15-82
    but that file was refactored and those lines now contain
    completely different code. The decision may still be valid
    (the logic moved) or may be obsolete.

    We measure staleness by content hash comparison, not just
    line number drift.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.MANUAL_FIX


class OrphanedDecisionError(DecisionError):
    """Decision links to files/functions that no longer exist.

    Edge case: file was deleted or function was removed during refactor.
    The decision might still carry useful architectural context even
    though its code anchor is gone.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.MANUAL_FIX


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------


class ContractError(CodebaseIntelError):
    """Base for quality contract failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "contracts"
        super().__init__(message, ctx)


class ContractViolationError(ContractError):
    """Code violates a quality contract.

    This is the most common "error" — it's really a signal, not a failure.
    The agent should see this as guidance, not a crash.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.MANUAL_FIX

    def __init__(
        self,
        contract_id: str,
        rule: str,
        violation_detail: str,
        context: ErrorContext | None = None,
    ) -> None:
        self.contract_id = contract_id
        self.rule = rule
        self.violation_detail = violation_detail
        message = f"[{contract_id}] {rule}: {violation_detail}"
        super().__init__(message, context)


class ContractConflictError(ContractError):
    """Two contracts impose contradictory requirements.

    Edge case: Contract A says "max function length 50 lines" and
    Contract B says "no helper functions for single-use logic."
    For complex logic, these conflict.

    Also: architectural rule says "no direct DB access outside repositories"
    but a performance contract says "use raw SQL for bulk operations."

    Resolution: contracts have priority levels. Higher-priority wins.
    If same priority, flag as conflict.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.MANUAL_FIX


class ContractParseError(ContractError):
    """Contract definition file has invalid syntax.

    Edge case: user hand-edited a YAML contract file and introduced
    a syntax error. Must not crash — report which contract is broken
    and continue evaluating the rest.
    """

    severity = Severity.ERROR
    recovery = RecoveryHint.UPDATE_CONFIG


# ---------------------------------------------------------------------------
# Orchestrator errors
# ---------------------------------------------------------------------------


class OrchestratorError(CodebaseIntelError):
    """Base for context assembly failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "orchestrator"
        super().__init__(message, ctx)


class BudgetExceededError(OrchestratorError):
    """Relevant context exceeds the agent's token budget.

    Edge case: task touches a core module that 200 files depend on.
    Even the minimum relevant context is 50K tokens but the agent
    only has 8K budget.

    Recovery: orchestrator must prioritize ruthlessly — closest
    dependencies first, most recent decisions first, highest-priority
    contracts first. Return what fits + a "truncated" flag.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.REDUCE_SCOPE

    def __init__(
        self,
        budget_tokens: int,
        required_tokens: int,
        context: ErrorContext | None = None,
    ) -> None:
        self.budget_tokens = budget_tokens
        self.required_tokens = required_tokens
        message = (
            f"Context requires ~{required_tokens} tokens "
            f"but budget is {budget_tokens}"
        )
        super().__init__(message, context)


class PartialInitializationError(OrchestratorError):
    """Some components are initialized but not others.

    Edge case: user ran `init` but it crashed halfway — graph exists
    but decisions database doesn't. The orchestrator must work with
    whatever is available and clearly indicate what's missing.
    """

    severity = Severity.WARNING
    recovery = RecoveryHint.REINITIALIZE

    def __init__(
        self,
        available: list[str],
        missing: list[str],
        context: ErrorContext | None = None,
    ) -> None:
        self.available = available
        self.missing = missing
        message = (
            f"Partial init — available: {available}, missing: {missing}"
        )
        super().__init__(message, context)


# ---------------------------------------------------------------------------
# Drift errors
# ---------------------------------------------------------------------------


class DriftError(CodebaseIntelError):
    """Base for drift detection failures."""

    def __init__(self, message: str, context: ErrorContext | None = None) -> None:
        ctx = context or ErrorContext()
        ctx.component = ctx.component or "drift"
        super().__init__(message, ctx)


class ContextRotError(DriftError):
    """Context records have drifted significantly from actual code.

    Edge case: after a major refactor, 40% of decision records
    reference moved/renamed files. The system detects this level
    of drift and flags it as a "context rot event" requiring
    bulk review, rather than 200 individual warnings.
    """

    severity = Severity.ERROR
    recovery = RecoveryHint.REINITIALIZE

    def __init__(
        self,
        rot_percentage: float,
        stale_records: int,
        total_records: int,
        context: ErrorContext | None = None,
    ) -> None:
        self.rot_percentage = rot_percentage
        self.stale_records = stale_records
        self.total_records = total_records
        message = (
            f"Context rot: {rot_percentage:.0%} of records stale "
            f"({stale_records}/{total_records})"
        )
        super().__init__(message, context)
