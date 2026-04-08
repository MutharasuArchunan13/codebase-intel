"""Tests for codebase_intel.core.exceptions module.

Covers the full exception hierarchy: base classes, error context,
serialization, severity/recovery enum values, and component auto-population.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codebase_intel.core.exceptions import (
    BudgetExceededError,
    CircularDependencyError,
    CodebaseIntelError,
    ContextRotError,
    ContractConflictError,
    ContractError,
    ContractParseError,
    ContractViolationError,
    DecisionConflictError,
    DecisionError,
    DriftError,
    ErrorContext,
    GraphError,
    OrchestratorError,
    OrphanedDecisionError,
    ParseError,
    PartialInitializationError,
    RecoveryHint,
    Severity,
    StaleDecisionError,
    StorageConcurrencyError,
    StorageCorruptError,
    StorageError,
    StorageMigrationError,
    UnsupportedLanguageError,
)

# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_warning_value(self) -> None:
        assert Severity.WARNING.value == "warning"

    def test_error_value(self) -> None:
        assert Severity.ERROR.value == "error"

    def test_fatal_value(self) -> None:
        assert Severity.FATAL.value == "fatal"

    def test_all_members(self) -> None:
        members = {member.name for member in Severity}
        assert members == {"WARNING", "ERROR", "FATAL"}


# ---------------------------------------------------------------------------
# RecoveryHint enum
# ---------------------------------------------------------------------------


class TestRecoveryHint:
    def test_retry_value(self) -> None:
        assert RecoveryHint.RETRY.value == "retry"

    def test_reinitialize_value(self) -> None:
        assert RecoveryHint.REINITIALIZE.value == "reinitialize"

    def test_reduce_scope_value(self) -> None:
        assert RecoveryHint.REDUCE_SCOPE.value == "reduce_scope"

    def test_manual_fix_value(self) -> None:
        assert RecoveryHint.MANUAL_FIX.value == "manual_fix"

    def test_skip_value(self) -> None:
        assert RecoveryHint.SKIP.value == "skip"

    def test_update_config_value(self) -> None:
        assert RecoveryHint.UPDATE_CONFIG.value == "update_config"

    def test_all_members(self) -> None:
        members = {member.name for member in RecoveryHint}
        assert members == {
            "RETRY",
            "REINITIALIZE",
            "REDUCE_SCOPE",
            "MANUAL_FIX",
            "SKIP",
            "UPDATE_CONFIG",
        }


# ---------------------------------------------------------------------------
# ErrorContext
# ---------------------------------------------------------------------------


class TestErrorContext:
    def test_default_values(self) -> None:
        ctx = ErrorContext()
        assert ctx.file_path is None
        assert ctx.line_range is None
        assert ctx.component == ""
        assert ctx.operation == ""
        assert ctx.details == {}

    def test_file_path_accepts_path_object(self) -> None:
        ctx = ErrorContext(file_path=Path("/src/main.py"))
        assert ctx.file_path == Path("/src/main.py")

    def test_line_range_tuple(self) -> None:
        ctx = ErrorContext(line_range=(10, 50))
        assert ctx.line_range == (10, 50)

    def test_component_and_operation(self) -> None:
        ctx = ErrorContext(component="graph", operation="parse_file")
        assert ctx.component == "graph"
        assert ctx.operation == "parse_file"

    def test_details_dict(self) -> None:
        details = {"language": "python", "file_size": 1024}
        ctx = ErrorContext(details=details)
        assert ctx.details == details

    def test_all_fields_populated(self) -> None:
        ctx = ErrorContext(
            file_path=Path("/src/auth.py"),
            line_range=(1, 100),
            component="contracts",
            operation="evaluate",
            details={"rule": "max-complexity"},
        )
        assert ctx.file_path == Path("/src/auth.py")
        assert ctx.line_range == (1, 100)
        assert ctx.component == "contracts"
        assert ctx.operation == "evaluate"
        assert ctx.details == {"rule": "max-complexity"}

    def test_details_default_is_independent_per_instance(self) -> None:
        """Each ErrorContext instance gets its own details dict (no shared mutable default)."""
        ctx_a = ErrorContext()
        ctx_b = ErrorContext()
        ctx_a.details["key"] = "value"
        assert "key" not in ctx_b.details


# ---------------------------------------------------------------------------
# CodebaseIntelError (base)
# ---------------------------------------------------------------------------


class TestCodebaseIntelError:
    def test_message_stored(self) -> None:
        error = CodebaseIntelError("something broke")
        assert str(error) == "something broke"

    def test_is_exception(self) -> None:
        error = CodebaseIntelError("fail")
        assert isinstance(error, Exception)

    def test_default_severity(self) -> None:
        error = CodebaseIntelError("fail")
        assert error.severity == Severity.ERROR

    def test_default_recovery(self) -> None:
        error = CodebaseIntelError("fail")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_default_context_created_when_none(self) -> None:
        error = CodebaseIntelError("fail")
        assert isinstance(error.context, ErrorContext)
        assert error.context.file_path is None
        assert error.context.component == ""

    def test_custom_context_preserved(self) -> None:
        ctx = ErrorContext(
            file_path=Path("/foo.py"),
            component="test",
            operation="validate",
        )
        error = CodebaseIntelError("fail", context=ctx)
        assert error.context is ctx
        assert error.context.file_path == Path("/foo.py")

    def test_to_dict_structure(self) -> None:
        error = CodebaseIntelError("test error")
        result = error.to_dict()
        assert result["error"] == "CodebaseIntelError"
        assert result["message"] == "test error"
        assert result["severity"] == "error"
        assert result["recovery"] == "manual_fix"
        assert "context" in result
        assert result["context"]["file_path"] is None
        assert result["context"]["line_range"] is None
        assert result["context"]["component"] == ""
        assert result["context"]["operation"] == ""
        assert result["context"]["details"] == {}

    def test_to_dict_with_full_context(self) -> None:
        ctx = ErrorContext(
            file_path=Path("/src/app.py"),
            line_range=(10, 20),
            component="graph",
            operation="build",
            details={"nodes": 42},
        )
        error = CodebaseIntelError("fail", context=ctx)
        result = error.to_dict()
        assert result["context"]["file_path"] == "/src/app.py"
        assert result["context"]["line_range"] == (10, 20)
        assert result["context"]["component"] == "graph"
        assert result["context"]["operation"] == "build"
        assert result["context"]["details"] == {"nodes": 42}

    def test_to_dict_is_json_serializable(self) -> None:
        ctx = ErrorContext(
            file_path=Path("/src/main.py"),
            line_range=(1, 50),
            component="test",
            operation="verify",
            details={"count": 5, "names": ["a", "b"]},
        )
        error = CodebaseIntelError("json test", context=ctx)
        result = error.to_dict()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["error"] == "CodebaseIntelError"
        assert deserialized["message"] == "json test"

    def test_to_dict_error_name_matches_subclass(self) -> None:
        """Subclass name appears in to_dict, not the base class name."""
        error = StorageCorruptError("corrupt db")
        result = error.to_dict()
        assert result["error"] == "StorageCorruptError"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(CodebaseIntelError, match="boom"):
            raise CodebaseIntelError("boom")


# ---------------------------------------------------------------------------
# Storage errors
# ---------------------------------------------------------------------------


class TestStorageError:
    def test_component_auto_set(self) -> None:
        error = StorageError("disk fail")
        assert error.context.component == "storage"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-storage")
        error = StorageError("fail", context=ctx)
        assert error.context.component == "custom-storage"

    def test_inherits_from_base(self) -> None:
        error = StorageError("fail")
        assert isinstance(error, CodebaseIntelError)

    def test_default_severity_and_recovery(self) -> None:
        error = StorageError("fail")
        assert error.severity == Severity.ERROR
        assert error.recovery == RecoveryHint.MANUAL_FIX


class TestStorageCorruptError:
    def test_severity(self) -> None:
        error = StorageCorruptError("corrupt")
        assert error.severity == Severity.ERROR

    def test_recovery(self) -> None:
        error = StorageCorruptError("corrupt")
        assert error.recovery == RecoveryHint.REINITIALIZE

    def test_component_auto_set(self) -> None:
        error = StorageCorruptError("corrupt")
        assert error.context.component == "storage"

    def test_inherits_from_storage_error(self) -> None:
        error = StorageCorruptError("corrupt")
        assert isinstance(error, StorageError)
        assert isinstance(error, CodebaseIntelError)

    def test_to_dict(self) -> None:
        error = StorageCorruptError("db is corrupted")
        result = error.to_dict()
        assert result["error"] == "StorageCorruptError"
        assert result["severity"] == "error"
        assert result["recovery"] == "reinitialize"
        assert result["context"]["component"] == "storage"

    def test_to_dict_json_serializable(self) -> None:
        error = StorageCorruptError("corrupt file")
        json.dumps(error.to_dict())


class TestStorageConcurrencyError:
    def test_severity(self) -> None:
        error = StorageConcurrencyError("lock conflict")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = StorageConcurrencyError("lock conflict")
        assert error.recovery == RecoveryHint.RETRY

    def test_component_auto_set(self) -> None:
        error = StorageConcurrencyError("lock conflict")
        assert error.context.component == "storage"

    def test_to_dict(self) -> None:
        error = StorageConcurrencyError("concurrent write")
        result = error.to_dict()
        assert result["error"] == "StorageConcurrencyError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "retry"


class TestStorageMigrationError:
    def test_severity(self) -> None:
        error = StorageMigrationError("schema mismatch")
        assert error.severity == Severity.ERROR

    def test_recovery(self) -> None:
        error = StorageMigrationError("schema mismatch")
        assert error.recovery == RecoveryHint.REINITIALIZE

    def test_component_auto_set(self) -> None:
        error = StorageMigrationError("schema mismatch")
        assert error.context.component == "storage"

    def test_to_dict(self) -> None:
        error = StorageMigrationError("version 3 expected, got 1")
        result = error.to_dict()
        assert result["error"] == "StorageMigrationError"
        assert result["severity"] == "error"
        assert result["recovery"] == "reinitialize"


# ---------------------------------------------------------------------------
# Graph errors
# ---------------------------------------------------------------------------


class TestGraphError:
    def test_component_auto_set(self) -> None:
        error = GraphError("graph fail")
        assert error.context.component == "graph"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-graph")
        error = GraphError("fail", context=ctx)
        assert error.context.component == "custom-graph"

    def test_inherits_from_base(self) -> None:
        error = GraphError("fail")
        assert isinstance(error, CodebaseIntelError)


class TestParseError:
    def test_severity(self) -> None:
        error = ParseError("parse failed")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = ParseError("parse failed")
        assert error.recovery == RecoveryHint.SKIP

    def test_component_auto_set(self) -> None:
        error = ParseError("parse failed")
        assert error.context.component == "graph"

    def test_inherits_from_graph_error(self) -> None:
        error = ParseError("parse failed")
        assert isinstance(error, GraphError)
        assert isinstance(error, CodebaseIntelError)

    def test_to_dict(self) -> None:
        ctx = ErrorContext(file_path=Path("/src/broken.py"))
        error = ParseError("syntax error", context=ctx)
        result = error.to_dict()
        assert result["error"] == "ParseError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "skip"
        assert result["context"]["file_path"] == "/src/broken.py"
        assert result["context"]["component"] == "graph"

    def test_to_dict_json_serializable(self) -> None:
        error = ParseError("binary file")
        json.dumps(error.to_dict())


class TestCircularDependencyError:
    def test_cycle_path_stored(self) -> None:
        error = CircularDependencyError(["A", "B", "C", "A"])
        assert error.cycle_path == ["A", "B", "C", "A"]

    def test_message_format(self) -> None:
        error = CircularDependencyError(["auth", "users", "auth"])
        assert str(error) == "Circular dependency: auth \u2192 users \u2192 auth"

    def test_severity(self) -> None:
        error = CircularDependencyError(["A", "B", "A"])
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = CircularDependencyError(["A", "B", "A"])
        assert error.recovery == RecoveryHint.SKIP

    def test_component_auto_set(self) -> None:
        error = CircularDependencyError(["A", "B", "A"])
        assert error.context.component == "graph"

    def test_with_context(self) -> None:
        ctx = ErrorContext(operation="dependency_analysis")
        error = CircularDependencyError(["X", "Y", "Z", "X"], context=ctx)
        assert error.context.operation == "dependency_analysis"
        assert error.context.component == "graph"

    def test_to_dict(self) -> None:
        error = CircularDependencyError(["mod_a", "mod_b", "mod_a"])
        result = error.to_dict()
        assert result["error"] == "CircularDependencyError"
        assert "mod_a" in result["message"]
        assert result["severity"] == "warning"

    def test_to_dict_json_serializable(self) -> None:
        error = CircularDependencyError(["A", "B", "C", "A"])
        json.dumps(error.to_dict())

    def test_single_node_cycle(self) -> None:
        error = CircularDependencyError(["A", "A"])
        assert str(error) == "Circular dependency: A \u2192 A"
        assert error.cycle_path == ["A", "A"]

    def test_empty_cycle_path(self) -> None:
        error = CircularDependencyError([])
        assert str(error) == "Circular dependency: "
        assert error.cycle_path == []


class TestUnsupportedLanguageError:
    def test_severity(self) -> None:
        error = UnsupportedLanguageError("no grammar for .proto")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = UnsupportedLanguageError("no grammar for .proto")
        assert error.recovery == RecoveryHint.SKIP

    def test_component_auto_set(self) -> None:
        error = UnsupportedLanguageError("no grammar")
        assert error.context.component == "graph"

    def test_inherits_from_graph_error(self) -> None:
        error = UnsupportedLanguageError("unsupported")
        assert isinstance(error, GraphError)

    def test_to_dict(self) -> None:
        error = UnsupportedLanguageError(".graphql not supported")
        result = error.to_dict()
        assert result["error"] == "UnsupportedLanguageError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "skip"


# ---------------------------------------------------------------------------
# Decision errors
# ---------------------------------------------------------------------------


class TestDecisionError:
    def test_component_auto_set(self) -> None:
        error = DecisionError("decision fail")
        assert error.context.component == "decisions"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-decisions")
        error = DecisionError("fail", context=ctx)
        assert error.context.component == "custom-decisions"

    def test_inherits_from_base(self) -> None:
        error = DecisionError("fail")
        assert isinstance(error, CodebaseIntelError)


class TestDecisionConflictError:
    def test_decision_ids_stored(self) -> None:
        error = DecisionConflictError("DEC-042", "DEC-058")
        assert error.decision_a == "DEC-042"
        assert error.decision_b == "DEC-058"

    def test_message_format(self) -> None:
        error = DecisionConflictError("DEC-001", "DEC-002")
        assert str(error) == "Conflicting decisions: DEC-001 vs DEC-002"

    def test_severity(self) -> None:
        error = DecisionConflictError("DEC-001", "DEC-002")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = DecisionConflictError("DEC-001", "DEC-002")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_component_auto_set(self) -> None:
        error = DecisionConflictError("DEC-001", "DEC-002")
        assert error.context.component == "decisions"

    def test_with_context(self) -> None:
        ctx = ErrorContext(operation="conflict_detection")
        error = DecisionConflictError("DEC-010", "DEC-020", context=ctx)
        assert error.context.operation == "conflict_detection"
        assert error.context.component == "decisions"

    def test_to_dict(self) -> None:
        error = DecisionConflictError("DEC-042", "DEC-058")
        result = error.to_dict()
        assert result["error"] == "DecisionConflictError"
        assert "DEC-042" in result["message"]
        assert "DEC-058" in result["message"]
        assert result["severity"] == "warning"
        assert result["recovery"] == "manual_fix"
        assert result["context"]["component"] == "decisions"

    def test_to_dict_json_serializable(self) -> None:
        error = DecisionConflictError("DEC-001", "DEC-002")
        json.dumps(error.to_dict())

    def test_inherits_from_decision_error(self) -> None:
        error = DecisionConflictError("A", "B")
        assert isinstance(error, DecisionError)
        assert isinstance(error, CodebaseIntelError)


class TestStaleDecisionError:
    def test_severity(self) -> None:
        error = StaleDecisionError("decision outdated")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = StaleDecisionError("decision outdated")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_component_auto_set(self) -> None:
        error = StaleDecisionError("stale")
        assert error.context.component == "decisions"

    def test_inherits_from_decision_error(self) -> None:
        error = StaleDecisionError("stale")
        assert isinstance(error, DecisionError)

    def test_to_dict(self) -> None:
        error = StaleDecisionError("content hash mismatch")
        result = error.to_dict()
        assert result["error"] == "StaleDecisionError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "manual_fix"
        assert result["context"]["component"] == "decisions"


class TestOrphanedDecisionError:
    def test_severity(self) -> None:
        error = OrphanedDecisionError("file deleted")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = OrphanedDecisionError("file deleted")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_component_auto_set(self) -> None:
        error = OrphanedDecisionError("orphaned")
        assert error.context.component == "decisions"

    def test_inherits_from_decision_error(self) -> None:
        error = OrphanedDecisionError("orphaned")
        assert isinstance(error, DecisionError)

    def test_to_dict(self) -> None:
        error = OrphanedDecisionError("referenced function removed")
        result = error.to_dict()
        assert result["error"] == "OrphanedDecisionError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "manual_fix"


# ---------------------------------------------------------------------------
# Contract errors
# ---------------------------------------------------------------------------


class TestContractError:
    def test_component_auto_set(self) -> None:
        error = ContractError("contract fail")
        assert error.context.component == "contracts"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-contracts")
        error = ContractError("fail", context=ctx)
        assert error.context.component == "custom-contracts"

    def test_inherits_from_base(self) -> None:
        error = ContractError("fail")
        assert isinstance(error, CodebaseIntelError)


class TestContractViolationError:
    def test_fields_stored(self) -> None:
        error = ContractViolationError(
            contract_id="ARCH-001",
            rule="no-direct-db-access",
            violation_detail="Repository bypassed in user_handler.py",
        )
        assert error.contract_id == "ARCH-001"
        assert error.rule == "no-direct-db-access"
        assert error.violation_detail == "Repository bypassed in user_handler.py"

    def test_message_format(self) -> None:
        error = ContractViolationError(
            contract_id="PERF-003",
            rule="max-query-count",
            violation_detail="15 queries in single request",
        )
        assert str(error) == "[PERF-003] max-query-count: 15 queries in single request"

    def test_severity(self) -> None:
        error = ContractViolationError("C1", "rule1", "detail1")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = ContractViolationError("C1", "rule1", "detail1")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_component_auto_set(self) -> None:
        error = ContractViolationError("C1", "rule1", "detail1")
        assert error.context.component == "contracts"

    def test_with_context(self) -> None:
        ctx = ErrorContext(
            file_path=Path("/src/handlers/user.py"),
            line_range=(42, 55),
            operation="evaluate_contract",
        )
        error = ContractViolationError("ARCH-001", "rule", "detail", context=ctx)
        assert error.context.file_path == Path("/src/handlers/user.py")
        assert error.context.line_range == (42, 55)
        assert error.context.component == "contracts"

    def test_to_dict(self) -> None:
        error = ContractViolationError(
            contract_id="SEC-002",
            rule="no-eval",
            violation_detail="eval() used in template.py",
        )
        result = error.to_dict()
        assert result["error"] == "ContractViolationError"
        assert result["message"] == "[SEC-002] no-eval: eval() used in template.py"
        assert result["severity"] == "warning"
        assert result["recovery"] == "manual_fix"
        assert result["context"]["component"] == "contracts"

    def test_to_dict_json_serializable(self) -> None:
        error = ContractViolationError("C1", "rule", "detail")
        json.dumps(error.to_dict())

    def test_inherits_from_contract_error(self) -> None:
        error = ContractViolationError("C1", "r1", "d1")
        assert isinstance(error, ContractError)
        assert isinstance(error, CodebaseIntelError)


class TestContractConflictError:
    def test_severity(self) -> None:
        error = ContractConflictError("conflicting contracts")
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = ContractConflictError("conflicting contracts")
        assert error.recovery == RecoveryHint.MANUAL_FIX

    def test_component_auto_set(self) -> None:
        error = ContractConflictError("conflict")
        assert error.context.component == "contracts"

    def test_inherits_from_contract_error(self) -> None:
        error = ContractConflictError("conflict")
        assert isinstance(error, ContractError)

    def test_to_dict(self) -> None:
        error = ContractConflictError("max-lines vs no-helpers conflict")
        result = error.to_dict()
        assert result["error"] == "ContractConflictError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "manual_fix"


class TestContractParseError:
    def test_severity(self) -> None:
        error = ContractParseError("YAML syntax error")
        assert error.severity == Severity.ERROR

    def test_recovery(self) -> None:
        error = ContractParseError("YAML syntax error")
        assert error.recovery == RecoveryHint.UPDATE_CONFIG

    def test_component_auto_set(self) -> None:
        error = ContractParseError("bad yaml")
        assert error.context.component == "contracts"

    def test_inherits_from_contract_error(self) -> None:
        error = ContractParseError("parse fail")
        assert isinstance(error, ContractError)

    def test_to_dict(self) -> None:
        error = ContractParseError("invalid contract definition")
        result = error.to_dict()
        assert result["error"] == "ContractParseError"
        assert result["severity"] == "error"
        assert result["recovery"] == "update_config"


# ---------------------------------------------------------------------------
# Orchestrator errors
# ---------------------------------------------------------------------------


class TestOrchestratorError:
    def test_component_auto_set(self) -> None:
        error = OrchestratorError("orchestrator fail")
        assert error.context.component == "orchestrator"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-orchestrator")
        error = OrchestratorError("fail", context=ctx)
        assert error.context.component == "custom-orchestrator"

    def test_inherits_from_base(self) -> None:
        error = OrchestratorError("fail")
        assert isinstance(error, CodebaseIntelError)


class TestBudgetExceededError:
    def test_token_fields_stored(self) -> None:
        error = BudgetExceededError(budget_tokens=8000, required_tokens=50000)
        assert error.budget_tokens == 8000
        assert error.required_tokens == 50000

    def test_message_format(self) -> None:
        error = BudgetExceededError(budget_tokens=8000, required_tokens=50000)
        assert str(error) == "Context requires ~50000 tokens but budget is 8000"

    def test_severity(self) -> None:
        error = BudgetExceededError(budget_tokens=100, required_tokens=200)
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = BudgetExceededError(budget_tokens=100, required_tokens=200)
        assert error.recovery == RecoveryHint.REDUCE_SCOPE

    def test_component_auto_set(self) -> None:
        error = BudgetExceededError(budget_tokens=100, required_tokens=200)
        assert error.context.component == "orchestrator"

    def test_with_context(self) -> None:
        ctx = ErrorContext(operation="assemble_context")
        error = BudgetExceededError(budget_tokens=4000, required_tokens=12000, context=ctx)
        assert error.context.operation == "assemble_context"
        assert error.context.component == "orchestrator"

    def test_to_dict(self) -> None:
        error = BudgetExceededError(budget_tokens=8000, required_tokens=50000)
        result = error.to_dict()
        assert result["error"] == "BudgetExceededError"
        assert "50000" in result["message"]
        assert "8000" in result["message"]
        assert result["severity"] == "warning"
        assert result["recovery"] == "reduce_scope"
        assert result["context"]["component"] == "orchestrator"

    def test_to_dict_json_serializable(self) -> None:
        error = BudgetExceededError(budget_tokens=1000, required_tokens=5000)
        json.dumps(error.to_dict())

    def test_inherits_from_orchestrator_error(self) -> None:
        error = BudgetExceededError(budget_tokens=100, required_tokens=200)
        assert isinstance(error, OrchestratorError)
        assert isinstance(error, CodebaseIntelError)

    def test_zero_budget(self) -> None:
        error = BudgetExceededError(budget_tokens=0, required_tokens=100)
        assert error.budget_tokens == 0
        assert "0" in str(error)

    def test_equal_budget_and_required(self) -> None:
        error = BudgetExceededError(budget_tokens=5000, required_tokens=5000)
        assert error.budget_tokens == 5000
        assert error.required_tokens == 5000


class TestPartialInitializationError:
    def test_lists_stored(self) -> None:
        error = PartialInitializationError(
            available=["graph", "contracts"],
            missing=["decisions"],
        )
        assert error.available == ["graph", "contracts"]
        assert error.missing == ["decisions"]

    def test_message_format(self) -> None:
        error = PartialInitializationError(
            available=["graph"],
            missing=["decisions", "contracts"],
        )
        expected_msg = (
            "Partial init \u2014 available: ['graph'], missing: ['decisions', 'contracts']"
        )
        assert str(error) == expected_msg

    def test_severity(self) -> None:
        error = PartialInitializationError(available=["graph"], missing=["decisions"])
        assert error.severity == Severity.WARNING

    def test_recovery(self) -> None:
        error = PartialInitializationError(available=["graph"], missing=["decisions"])
        assert error.recovery == RecoveryHint.REINITIALIZE

    def test_component_auto_set(self) -> None:
        error = PartialInitializationError(available=[], missing=["all"])
        assert error.context.component == "orchestrator"

    def test_with_context(self) -> None:
        ctx = ErrorContext(operation="init_check")
        error = PartialInitializationError(available=["graph"], missing=["decisions"], context=ctx)
        assert error.context.operation == "init_check"
        assert error.context.component == "orchestrator"

    def test_to_dict(self) -> None:
        error = PartialInitializationError(
            available=["graph", "contracts"],
            missing=["decisions"],
        )
        result = error.to_dict()
        assert result["error"] == "PartialInitializationError"
        assert result["severity"] == "warning"
        assert result["recovery"] == "reinitialize"
        assert result["context"]["component"] == "orchestrator"

    def test_to_dict_json_serializable(self) -> None:
        error = PartialInitializationError(available=["a"], missing=["b"])
        json.dumps(error.to_dict())

    def test_inherits_from_orchestrator_error(self) -> None:
        error = PartialInitializationError(available=[], missing=[])
        assert isinstance(error, OrchestratorError)
        assert isinstance(error, CodebaseIntelError)

    def test_empty_lists(self) -> None:
        error = PartialInitializationError(available=[], missing=[])
        assert error.available == []
        assert error.missing == []

    def test_all_missing(self) -> None:
        error = PartialInitializationError(
            available=[],
            missing=["graph", "decisions", "contracts"],
        )
        assert error.available == []
        assert len(error.missing) == 3


# ---------------------------------------------------------------------------
# Drift errors
# ---------------------------------------------------------------------------


class TestDriftError:
    def test_component_auto_set(self) -> None:
        error = DriftError("drift detected")
        assert error.context.component == "drift"

    def test_component_not_overridden_when_provided(self) -> None:
        ctx = ErrorContext(component="custom-drift")
        error = DriftError("fail", context=ctx)
        assert error.context.component == "custom-drift"

    def test_inherits_from_base(self) -> None:
        error = DriftError("fail")
        assert isinstance(error, CodebaseIntelError)


class TestContextRotError:
    def test_fields_stored(self) -> None:
        error = ContextRotError(
            rot_percentage=0.4,
            stale_records=80,
            total_records=200,
        )
        assert error.rot_percentage == 0.4
        assert error.stale_records == 80
        assert error.total_records == 200

    def test_message_format(self) -> None:
        error = ContextRotError(
            rot_percentage=0.4,
            stale_records=80,
            total_records=200,
        )
        assert str(error) == "Context rot: 40% of records stale (80/200)"

    def test_message_format_high_percentage(self) -> None:
        error = ContextRotError(
            rot_percentage=0.95,
            stale_records=190,
            total_records=200,
        )
        assert str(error) == "Context rot: 95% of records stale (190/200)"

    def test_message_format_zero_percentage(self) -> None:
        error = ContextRotError(
            rot_percentage=0.0,
            stale_records=0,
            total_records=100,
        )
        assert str(error) == "Context rot: 0% of records stale (0/100)"

    def test_message_format_full_rot(self) -> None:
        error = ContextRotError(
            rot_percentage=1.0,
            stale_records=50,
            total_records=50,
        )
        assert str(error) == "Context rot: 100% of records stale (50/50)"

    def test_severity(self) -> None:
        error = ContextRotError(rot_percentage=0.5, stale_records=50, total_records=100)
        assert error.severity == Severity.ERROR

    def test_recovery(self) -> None:
        error = ContextRotError(rot_percentage=0.5, stale_records=50, total_records=100)
        assert error.recovery == RecoveryHint.REINITIALIZE

    def test_component_auto_set(self) -> None:
        error = ContextRotError(rot_percentage=0.1, stale_records=1, total_records=10)
        assert error.context.component == "drift"

    def test_with_context(self) -> None:
        ctx = ErrorContext(operation="drift_scan")
        error = ContextRotError(
            rot_percentage=0.3,
            stale_records=30,
            total_records=100,
            context=ctx,
        )
        assert error.context.operation == "drift_scan"
        assert error.context.component == "drift"

    def test_to_dict(self) -> None:
        error = ContextRotError(
            rot_percentage=0.4,
            stale_records=80,
            total_records=200,
        )
        result = error.to_dict()
        assert result["error"] == "ContextRotError"
        assert "40%" in result["message"]
        assert "80/200" in result["message"]
        assert result["severity"] == "error"
        assert result["recovery"] == "reinitialize"
        assert result["context"]["component"] == "drift"

    def test_to_dict_json_serializable(self) -> None:
        error = ContextRotError(rot_percentage=0.5, stale_records=50, total_records=100)
        json.dumps(error.to_dict())

    def test_inherits_from_drift_error(self) -> None:
        error = ContextRotError(rot_percentage=0.1, stale_records=1, total_records=10)
        assert isinstance(error, DriftError)
        assert isinstance(error, CodebaseIntelError)


# ---------------------------------------------------------------------------
# Cross-cutting: hierarchy & catch-all behavior
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Verify that catching a base class catches all subclasses."""

    def test_catch_codebase_intel_error_catches_storage(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise StorageCorruptError("corrupt")

    def test_catch_codebase_intel_error_catches_graph(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise ParseError("bad parse")

    def test_catch_codebase_intel_error_catches_decision(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise DecisionConflictError("DEC-1", "DEC-2")

    def test_catch_codebase_intel_error_catches_contract(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise ContractViolationError("C1", "rule", "detail")

    def test_catch_codebase_intel_error_catches_orchestrator(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise BudgetExceededError(100, 200)

    def test_catch_codebase_intel_error_catches_drift(self) -> None:
        with pytest.raises(CodebaseIntelError):
            raise ContextRotError(0.5, 50, 100)

    def test_catch_storage_error_catches_subclasses(self) -> None:
        for exc_class in (StorageCorruptError, StorageConcurrencyError, StorageMigrationError):
            with pytest.raises(StorageError):
                raise exc_class("test")

    def test_catch_graph_error_catches_subclasses(self) -> None:
        with pytest.raises(GraphError):
            raise ParseError("test")
        with pytest.raises(GraphError):
            raise CircularDependencyError(["A", "B", "A"])
        with pytest.raises(GraphError):
            raise UnsupportedLanguageError("test")

    def test_catch_decision_error_catches_subclasses(self) -> None:
        for exc_class in (StaleDecisionError, OrphanedDecisionError):
            with pytest.raises(DecisionError):
                raise exc_class("test")
        with pytest.raises(DecisionError):
            raise DecisionConflictError("A", "B")

    def test_catch_contract_error_catches_subclasses(self) -> None:
        with pytest.raises(ContractError):
            raise ContractViolationError("C", "r", "d")
        for exc_class in (ContractConflictError, ContractParseError):
            with pytest.raises(ContractError):
                raise exc_class("test")

    def test_catch_orchestrator_error_catches_subclasses(self) -> None:
        with pytest.raises(OrchestratorError):
            raise BudgetExceededError(100, 200)
        with pytest.raises(OrchestratorError):
            raise PartialInitializationError(["a"], ["b"])

    def test_catch_drift_error_catches_subclasses(self) -> None:
        with pytest.raises(DriftError):
            raise ContextRotError(0.1, 1, 10)


# ---------------------------------------------------------------------------
# Cross-cutting: to_dict JSON serializability across all concrete exceptions
# ---------------------------------------------------------------------------


class TestToDictJsonSerializability:
    """Every concrete exception must produce a JSON-serializable dict."""

    @pytest.fixture(
        params=[
            lambda: CodebaseIntelError("base error"),
            lambda: StorageError("storage"),
            lambda: StorageCorruptError("corrupt"),
            lambda: StorageConcurrencyError("concurrency"),
            lambda: StorageMigrationError("migration"),
            lambda: GraphError("graph"),
            lambda: ParseError("parse"),
            lambda: CircularDependencyError(["A", "B", "A"]),
            lambda: UnsupportedLanguageError("unsupported"),
            lambda: DecisionError("decision"),
            lambda: DecisionConflictError("DEC-1", "DEC-2"),
            lambda: StaleDecisionError("stale"),
            lambda: OrphanedDecisionError("orphaned"),
            lambda: ContractError("contract"),
            lambda: ContractViolationError("C1", "rule", "detail"),
            lambda: ContractConflictError("conflict"),
            lambda: ContractParseError("parse"),
            lambda: OrchestratorError("orchestrator"),
            lambda: BudgetExceededError(100, 200),
            lambda: PartialInitializationError(["a"], ["b"]),
            lambda: DriftError("drift"),
            lambda: ContextRotError(0.5, 50, 100),
        ],
        ids=[
            "CodebaseIntelError",
            "StorageError",
            "StorageCorruptError",
            "StorageConcurrencyError",
            "StorageMigrationError",
            "GraphError",
            "ParseError",
            "CircularDependencyError",
            "UnsupportedLanguageError",
            "DecisionError",
            "DecisionConflictError",
            "StaleDecisionError",
            "OrphanedDecisionError",
            "ContractError",
            "ContractViolationError",
            "ContractConflictError",
            "ContractParseError",
            "OrchestratorError",
            "BudgetExceededError",
            "PartialInitializationError",
            "DriftError",
            "ContextRotError",
        ],
    )
    def error_instance(self, request: pytest.FixtureRequest) -> CodebaseIntelError:
        return request.param()

    def test_to_dict_returns_dict(self, error_instance: CodebaseIntelError) -> None:
        result = error_instance.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_required_keys(self, error_instance: CodebaseIntelError) -> None:
        result = error_instance.to_dict()
        assert "error" in result
        assert "message" in result
        assert "severity" in result
        assert "recovery" in result
        assert "context" in result

    def test_to_dict_json_round_trip(self, error_instance: CodebaseIntelError) -> None:
        result = error_instance.to_dict()
        serialized = json.dumps(result)
        deserialized = json.loads(serialized)
        assert deserialized["error"] == result["error"]
        assert deserialized["message"] == result["message"]
        assert deserialized["severity"] == result["severity"]
        assert deserialized["recovery"] == result["recovery"]

    def test_to_dict_severity_is_valid_enum_value(self, error_instance: CodebaseIntelError) -> None:
        result = error_instance.to_dict()
        valid_values = {s.value for s in Severity}
        assert result["severity"] in valid_values

    def test_to_dict_recovery_is_valid_enum_value(self, error_instance: CodebaseIntelError) -> None:
        result = error_instance.to_dict()
        valid_values = {r.value for r in RecoveryHint}
        assert result["recovery"] in valid_values

    def test_to_dict_context_has_required_keys(self, error_instance: CodebaseIntelError) -> None:
        ctx = error_instance.to_dict()["context"]
        assert "file_path" in ctx
        assert "line_range" in ctx
        assert "component" in ctx
        assert "operation" in ctx
        assert "details" in ctx
