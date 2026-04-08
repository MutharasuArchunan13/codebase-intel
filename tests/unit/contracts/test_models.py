"""Unit tests for codebase_intel.contracts.models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from codebase_intel.contracts.models import (
    ContractRule,
    QualityContract,
    RuleKind,
    ScopeFilter,
    builtin_ai_guardrails,
    builtin_architecture_rules,
)
from codebase_intel.core.types import ContractSeverity, Language

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_scope(**overrides: object) -> ScopeFilter:
    defaults: dict[str, object] = {
        "include_patterns": [],
        "exclude_patterns": [],
        "languages": [],
        "exclude_tests": False,
        "exclude_generated": False,
    }
    defaults.update(overrides)
    return ScopeFilter(**defaults)  # type: ignore[arg-type]


def _make_rule(**overrides: object) -> ContractRule:
    defaults: dict[str, object] = {
        "id": "test-rule",
        "name": "Test Rule",
        "description": "A rule for testing",
        "kind": RuleKind.PATTERN,
        "severity": ContractSeverity.WARNING,
    }
    defaults.update(overrides)
    return ContractRule(**defaults)  # type: ignore[arg-type]


def _make_contract(**overrides: object) -> QualityContract:
    defaults: dict[str, object] = {
        "id": "test-contract",
        "name": "Test Contract",
        "description": "Contract for testing",
        "rules": [],
    }
    defaults.update(overrides)
    return QualityContract(**defaults)  # type: ignore[arg-type]


# ============================================================================
# ScopeFilter.matches
# ============================================================================


class TestScopeFilterMatches:
    """Covers include/exclude patterns, language filter, test/generated exclusion."""

    def test_empty_scope_matches_everything(self) -> None:
        scope = _make_scope()
        assert scope.matches(Path("src/foo.py")) is True
        assert scope.matches(Path("any/deep/path/bar.ts")) is True

    def test_include_pattern_restricts_matches(self) -> None:
        scope = _make_scope(include_patterns=["src/api/**"])
        assert scope.matches(Path("src/api/routes.py")) is True
        assert scope.matches(Path("src/core/types.py")) is False

    def test_exclude_pattern_removes_matches(self) -> None:
        scope = _make_scope(exclude_patterns=["*.min.js"])
        assert scope.matches(Path("app.min.js")) is False
        assert scope.matches(Path("app.js")) is True

    def test_include_and_exclude_combined(self) -> None:
        """Include first, then subtract excludes."""
        scope = _make_scope(
            include_patterns=["src/**"],
            exclude_patterns=["src/generated/**"],
        )
        assert scope.matches(Path("src/core/models.py")) is True
        assert scope.matches(Path("src/generated/proto.py")) is False
        assert scope.matches(Path("tests/test_foo.py")) is False

    def test_language_filter_restricts_to_specified_languages(self) -> None:
        scope = _make_scope(languages=[Language.PYTHON])
        assert scope.matches(Path("foo.py"), language=Language.PYTHON) is True
        assert scope.matches(Path("foo.ts"), language=Language.TYPESCRIPT) is False

    def test_language_filter_empty_allows_all(self) -> None:
        scope = _make_scope(languages=[])
        assert scope.matches(Path("foo.py"), language=Language.PYTHON) is True
        assert scope.matches(Path("foo.rs"), language=Language.RUST) is True

    def test_exclude_tests_flag(self) -> None:
        scope = _make_scope(exclude_tests=True)
        assert scope.matches(Path("test_foo.py"), is_test=True) is False
        assert scope.matches(Path("src/foo.py"), is_test=False) is True

    def test_exclude_generated_flag(self) -> None:
        scope = _make_scope(exclude_generated=True)
        assert scope.matches(Path("proto.py"), is_generated=True) is False
        assert scope.matches(Path("proto.py"), is_generated=False) is True

    def test_exclude_generated_default_is_true(self) -> None:
        scope = ScopeFilter()
        assert scope.matches(Path("foo.py"), is_generated=True) is False

    def test_multiple_include_patterns(self) -> None:
        scope = _make_scope(include_patterns=["src/api/*", "src/core/*"])
        assert scope.matches(Path("src/api/routes.py")) is True
        assert scope.matches(Path("src/core/types.py")) is True
        assert scope.matches(Path("src/cli/main.py")) is False

    def test_exclude_checked_before_include(self) -> None:
        """A file matching both include and exclude should be excluded."""
        scope = _make_scope(
            include_patterns=["src/**"],
            exclude_patterns=["src/vendor/**"],
        )
        assert scope.matches(Path("src/vendor/lib.py")) is False

    def test_default_exclude_patterns(self) -> None:
        """ScopeFilter with no explicit excludes still has sensible defaults."""
        scope = ScopeFilter()
        assert scope.matches(Path("node_modules/lodash/index.js")) is False
        assert scope.matches(Path("vendor/lib.py")) is False
        assert scope.matches(Path("dist/bundle.js")) is False
        assert scope.matches(Path("app.min.js")) is False


# ============================================================================
# ContractRule.effective_severity
# ============================================================================


class TestContractRuleEffectiveSeverity:
    """Covers severity adjustment with/without migration deadline."""

    def test_no_deadline_returns_base_severity(self) -> None:
        rule = _make_rule(severity=ContractSeverity.WARNING)
        assert rule.effective_severity == ContractSeverity.WARNING

    def test_future_deadline_returns_base_severity(self) -> None:
        future = datetime.now(UTC) + timedelta(days=30)
        rule = _make_rule(
            severity=ContractSeverity.WARNING,
            migration_deadline=future,
        )
        assert rule.effective_severity == ContractSeverity.WARNING

    def test_past_deadline_escalates_to_error(self) -> None:
        past = datetime.now(UTC) - timedelta(days=1)
        rule = _make_rule(
            severity=ContractSeverity.WARNING,
            migration_deadline=past,
        )
        assert rule.effective_severity == ContractSeverity.ERROR

    def test_info_severity_escalates_after_deadline(self) -> None:
        past = datetime.now(UTC) - timedelta(days=1)
        rule = _make_rule(
            severity=ContractSeverity.INFO,
            migration_deadline=past,
        )
        assert rule.effective_severity == ContractSeverity.ERROR

    def test_error_severity_stays_error_regardless(self) -> None:
        rule = _make_rule(severity=ContractSeverity.ERROR)
        assert rule.effective_severity == ContractSeverity.ERROR

    def test_naive_datetime_converted_to_utc(self) -> None:
        naive_past = datetime(2020, 1, 1)
        rule = _make_rule(
            severity=ContractSeverity.WARNING,
            migration_deadline=naive_past,
        )
        assert rule.migration_deadline is not None
        assert rule.migration_deadline.tzinfo is not None
        assert rule.effective_severity == ContractSeverity.ERROR


# ============================================================================
# QualityContract.rules_for_file
# ============================================================================


class TestQualityContractRulesForFile:
    """Covers scope matching at contract level for rule retrieval."""

    def test_matching_scope_returns_all_rules(self) -> None:
        rules = [
            _make_rule(id="rule-1"),
            _make_rule(id="rule-2"),
        ]
        contract = _make_contract(
            rules=rules,
            scope=_make_scope(include_patterns=["src/**"]),
        )
        result = contract.rules_for_file(Path("src/foo.py"))
        assert len(result) == 2
        assert {r.id for r in result} == {"rule-1", "rule-2"}

    def test_non_matching_scope_returns_empty(self) -> None:
        rules = [_make_rule(id="rule-1")]
        contract = _make_contract(
            rules=rules,
            scope=_make_scope(include_patterns=["src/api/**"]),
        )
        result = contract.rules_for_file(Path("tests/test_foo.py"))
        assert result == []

    def test_excluded_file_returns_empty(self) -> None:
        rules = [_make_rule(id="rule-1")]
        contract = _make_contract(
            rules=rules,
            scope=_make_scope(exclude_patterns=["*.generated.*"]),
        )
        result = contract.rules_for_file(Path("proto.generated.py"))
        assert result == []

    def test_language_filter_on_scope(self) -> None:
        rules = [_make_rule(id="py-rule")]
        contract = _make_contract(
            rules=rules,
            scope=_make_scope(languages=[Language.PYTHON]),
        )
        assert len(contract.rules_for_file(Path("a.py"), language=Language.PYTHON)) == 1
        assert len(contract.rules_for_file(Path("a.ts"), language=Language.TYPESCRIPT)) == 0

    def test_test_file_excluded(self) -> None:
        rules = [_make_rule(id="no-tests")]
        contract = _make_contract(
            rules=rules,
            scope=_make_scope(exclude_tests=True),
        )
        assert contract.rules_for_file(Path("test_foo.py"), is_test=True) == []
        assert len(contract.rules_for_file(Path("foo.py"), is_test=False)) == 1


# ============================================================================
# QualityContract.to_context_string
# ============================================================================


class TestQualityContractToContextString:
    """Covers compact and verbose serialization."""

    def test_compact_mode_includes_header_and_rule_names(self) -> None:
        rules = [_make_rule(id="r1", name="Rule One", fix_suggestion="Fix it")]
        contract = _make_contract(
            id="ctr-1",
            name="My Contract",
            description="Desc",
            rules=rules,
        )
        text = contract.to_context_string(verbose=False)
        assert "My Contract" in text
        assert "ctr-1" in text
        assert "Rule One" in text
        # Compact mode should not include fix suggestion
        assert "Fix it" not in text

    def test_verbose_mode_includes_fix_suggestions(self) -> None:
        rules = [_make_rule(id="r1", name="Rule One", fix_suggestion="Fix it")]
        contract = _make_contract(rules=rules)
        text = contract.to_context_string(verbose=True)
        assert "Fix: Fix it" in text

    def test_verbose_mode_includes_examples(self) -> None:
        from codebase_intel.contracts.models import PatternExample

        examples = [
            PatternExample(code="good()", is_approved=True, description="Use good"),
            PatternExample(code="bad()", is_approved=False, description="Avoid bad"),
        ]
        rules = [_make_rule(id="r1", examples=examples)]
        contract = _make_contract(rules=rules)
        text = contract.to_context_string(verbose=True)
        assert "DO: Use good" in text
        assert "DON'T: Avoid bad" in text

    def test_severity_badges(self) -> None:
        rules = [
            _make_rule(id="r-err", severity=ContractSeverity.ERROR),
            _make_rule(id="r-warn", severity=ContractSeverity.WARNING),
            _make_rule(id="r-info", severity=ContractSeverity.INFO),
        ]
        contract = _make_contract(rules=rules)
        text = contract.to_context_string()
        assert "[ERROR]" in text
        assert "[WARN]" in text
        assert "[INFO]" in text

    def test_priority_and_rule_count_in_header(self) -> None:
        rules = [_make_rule(id="r1"), _make_rule(id="r2")]
        contract = _make_contract(rules=rules, priority=500)
        text = contract.to_context_string()
        assert "Priority: 500" in text
        assert "Rules: 2" in text


# ============================================================================
# builtin_ai_guardrails
# ============================================================================


class TestBuiltinAiGuardrails:
    """Verify all 6 rules exist with correct IDs and attributes."""

    def test_returns_quality_contract(self) -> None:
        contract = builtin_ai_guardrails()
        assert isinstance(contract, QualityContract)
        assert contract.id == "ai-guardrails"
        assert contract.is_builtin is True

    def test_has_exactly_6_rules(self) -> None:
        contract = builtin_ai_guardrails()
        assert len(contract.rules) == 6

    def test_rule_ids(self) -> None:
        contract = builtin_ai_guardrails()
        expected_ids = {
            "no-hallucinated-imports",
            "no-over-abstraction",
            "no-unnecessary-error-handling",
            "no-restating-comments",
            "no-speculative-features",
            "no-excessive-logging",
        }
        actual_ids = {r.id for r in contract.rules}
        assert actual_ids == expected_ids

    def test_all_rules_are_ai_antipattern_kind(self) -> None:
        contract = builtin_ai_guardrails()
        for rule in contract.rules:
            assert rule.kind == RuleKind.AI_ANTIPATTERN, (
                f"Rule {rule.id} should be AI_ANTIPATTERN, got {rule.kind}"
            )

    def test_hallucinated_imports_is_error_severity(self) -> None:
        contract = builtin_ai_guardrails()
        rule_map = {r.id: r for r in contract.rules}
        assert rule_map["no-hallucinated-imports"].severity == ContractSeverity.ERROR

    def test_restating_comments_has_pattern(self) -> None:
        contract = builtin_ai_guardrails()
        rule_map = {r.id: r for r in contract.rules}
        rule = rule_map["no-restating-comments"]
        assert rule.pattern is not None
        assert rule.severity == ContractSeverity.INFO

    def test_priority_is_500(self) -> None:
        contract = builtin_ai_guardrails()
        assert contract.priority == 500

    def test_tags_include_ai(self) -> None:
        contract = builtin_ai_guardrails()
        assert "ai" in contract.tags


# ============================================================================
# builtin_architecture_rules
# ============================================================================


class TestBuiltinArchitectureRules:
    """Verify architecture rules exist with correct structure."""

    def test_returns_quality_contract(self) -> None:
        contract = builtin_architecture_rules()
        assert isinstance(contract, QualityContract)
        assert contract.id == "architecture-basics"
        assert contract.is_builtin is True

    def test_has_rules(self) -> None:
        contract = builtin_architecture_rules()
        assert len(contract.rules) >= 2

    def test_rule_ids(self) -> None:
        contract = builtin_architecture_rules()
        expected_ids = {"no-circular-imports", "no-god-files", "no-god-functions"}
        actual_ids = {r.id for r in contract.rules}
        assert actual_ids == expected_ids

    def test_god_files_threshold(self) -> None:
        contract = builtin_architecture_rules()
        rule_map = {r.id: r for r in contract.rules}
        god_files = rule_map["no-god-files"]
        assert god_files.kind == RuleKind.THRESHOLD
        assert god_files.threshold_metric == "max_lines"
        assert god_files.threshold_value == 500

    def test_god_functions_threshold(self) -> None:
        contract = builtin_architecture_rules()
        rule_map = {r.id: r for r in contract.rules}
        god_funcs = rule_map["no-god-functions"]
        assert god_funcs.kind == RuleKind.THRESHOLD
        assert god_funcs.threshold_metric == "max_function_lines"
        assert god_funcs.threshold_value == 50

    def test_priority_is_400(self) -> None:
        contract = builtin_architecture_rules()
        assert contract.priority == 400

    def test_circular_imports_is_architectural(self) -> None:
        contract = builtin_architecture_rules()
        rule_map = {r.id: r for r in contract.rules}
        assert rule_map["no-circular-imports"].kind == RuleKind.ARCHITECTURAL
