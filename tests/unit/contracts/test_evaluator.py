"""Unit tests for codebase_intel.contracts.evaluator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codebase_intel.contracts.evaluator import (
    ContractEvaluator,
    EvaluationReport,
    Violation,
)
from codebase_intel.contracts.models import (
    ContractRule,
    QualityContract,
    RuleKind,
    ScopeFilter,
)
from codebase_intel.core.types import (
    ContractSeverity,
    GraphNode,
    Language,
    LineRange,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


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


def _make_contract(
    contract_id: str = "test-contract",
    rules: list[ContractRule] | None = None,
    **overrides: object,
) -> QualityContract:
    defaults: dict[str, object] = {
        "id": contract_id,
        "name": "Test Contract",
        "description": "Contract for testing",
        "rules": rules or [],
        "scope": ScopeFilter(
            include_patterns=[],
            exclude_patterns=[],
            exclude_generated=False,
        ),
    }
    defaults.update(overrides)
    return QualityContract(**defaults)  # type: ignore[arg-type]


def _make_registry(contracts: list[QualityContract]) -> MagicMock:
    registry = MagicMock()
    registry.get_all.return_value = contracts
    return registry


def _make_evaluator(
    contracts: list[QualityContract] | None = None,
) -> ContractEvaluator:
    registry = _make_registry(contracts or [])
    return ContractEvaluator(registry=registry, storage=None)


def _make_graph_node(
    name: str = "my_func",
    kind: NodeKind = NodeKind.FUNCTION,
    start: int = 1,
    end: int = 10,
) -> GraphNode:
    return GraphNode(
        node_id=GraphNode.make_id(Path("/fake.py"), kind, name),
        kind=kind,
        name=name,
        qualified_name=f"module.{name}",
        file_path=Path("/fake.py"),
        line_range=LineRange(start=start, end=end),
        language=Language.PYTHON,
    )


# ============================================================================
# ContractEvaluator._evaluate_pattern_rule
# ============================================================================


class TestEvaluatePatternRule:
    """Covers regex matching, comment filtering, and anti_pattern."""

    def test_pattern_match_produces_violation(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"TODO|FIXME",
        )
        content = "x = 1  # TODO: fix this\ny = 2\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert len(violations) == 1
        assert violations[0].line == 1
        assert violations[0].matched_text == "TODO"

    def test_pattern_no_match_returns_empty(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"HACK")
        content = "clean_code = True\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert violations == []

    def test_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"TODO")
        content = "# TODO: this is a comment\nreal_code = 1\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert violations == []

    def test_js_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"TODO")
        content = "// TODO: js comment\nlet x = 1;\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.js", content
        )
        assert violations == []

    def test_block_comment_lines_are_skipped(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"TODO")
        content = "/* TODO: block comment */\n* TODO: continuation\ncode();\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.js", content
        )
        assert violations == []

    def test_multiple_matches_on_different_lines(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"FIXME")
        content = "a = FIXME\nb = ok\nc = FIXME\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert len(violations) == 2
        assert violations[0].line == 1
        assert violations[1].line == 3

    def test_anti_pattern_absent_produces_violation(self, tmp_path: Path) -> None:
        """anti_pattern means the pattern SHOULD be present; absence is a violation."""
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=None,
            anti_pattern=r"Copyright \d{4}",
        )
        content = "def main():\n    pass\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert len(violations) == 1
        assert "Required pattern not found" in violations[0].message

    def test_anti_pattern_present_produces_no_violation(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=None,
            anti_pattern=r"Copyright \d{4}",
        )
        content = "# Copyright 2024\ndef main():\n    pass\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert violations == []

    def test_both_pattern_and_anti_pattern(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"eval\(",
            anti_pattern=r"# nosec",
        )
        content = "result = eval('expr')\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        # One from pattern match, one from missing anti_pattern
        assert len(violations) == 2

    def test_severity_uses_effective_severity(self, tmp_path: Path) -> None:
        from datetime import UTC, datetime, timedelta

        past = datetime.now(UTC) - timedelta(days=1)
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"old_api",
            severity=ContractSeverity.WARNING,
            migration_deadline=past,
        )
        content = "old_api()\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert len(violations) == 1
        assert violations[0].severity == ContractSeverity.ERROR


# ============================================================================
# ContractEvaluator._evaluate_threshold_rule
# ============================================================================


class TestEvaluateThresholdRule:
    """Covers max_lines violation and max_function_lines."""

    def test_max_lines_under_threshold_no_violation(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_lines",
            threshold_value=100,
        )
        content = "\n".join(f"line {i}" for i in range(50))
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", content, []
        )
        assert violations == []

    def test_max_lines_over_threshold_violation(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_lines",
            threshold_value=10,
        )
        content = "\n".join(f"line {i}" for i in range(20))
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", content, []
        )
        assert len(violations) == 1
        assert "20 lines" in violations[0].message
        assert "max: 10" in violations[0].message

    def test_max_function_lines_under_threshold(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_function_lines",
            threshold_value=50,
        )
        node = _make_graph_node(start=1, end=30)
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "", [node]
        )
        assert violations == []

    def test_max_function_lines_over_threshold(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_function_lines",
            threshold_value=50,
        )
        node = _make_graph_node(name="big_func", start=1, end=80)
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "", [node]
        )
        assert len(violations) == 1
        assert "big_func" in violations[0].message
        assert "80 lines" in violations[0].message

    def test_multiple_functions_some_violating(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_function_lines",
            threshold_value=20,
        )
        small = _make_graph_node(name="small_func", start=1, end=10)
        big = _make_graph_node(name="big_func", start=15, end=60)
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "", [small, big]
        )
        assert len(violations) == 1
        assert "big_func" in violations[0].message

    def test_method_nodes_also_checked(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_function_lines",
            threshold_value=10,
        )
        method = _make_graph_node(
            name="MyClass.long_method",
            kind=NodeKind.METHOD,
            start=1,
            end=50,
        )
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "", [method]
        )
        assert len(violations) == 1

    def test_class_nodes_not_checked_for_function_lines(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="max_function_lines",
            threshold_value=10,
        )
        class_node = _make_graph_node(
            name="BigClass",
            kind=NodeKind.CLASS,
            start=1,
            end=200,
        )
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "", [class_node]
        )
        assert violations == []

    def test_missing_metric_returns_empty(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric=None,
            threshold_value=None,
        )
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "some content", []
        )
        assert violations == []

    def test_unknown_metric_returns_empty(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.THRESHOLD,
            threshold_metric="unknown_metric",
            threshold_value=10,
        )
        violations = evaluator._evaluate_threshold_rule(
            rule, "ctr", tmp_path / "foo.py", "content", []
        )
        assert violations == []


# ============================================================================
# EvaluationReport
# ============================================================================


class TestEvaluationReport:
    """Covers error_count, warning_count, has_blocking_violations."""

    def test_empty_report(self) -> None:
        report = EvaluationReport()
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.has_blocking_violations is False

    def test_error_count(self) -> None:
        report = EvaluationReport(
            violations=[
                Violation(
                    contract_id="c",
                    rule_id="r1",
                    rule_name="R1",
                    severity=ContractSeverity.ERROR,
                    file_path=Path("a.py"),
                ),
                Violation(
                    contract_id="c",
                    rule_id="r2",
                    rule_name="R2",
                    severity=ContractSeverity.WARNING,
                    file_path=Path("a.py"),
                ),
                Violation(
                    contract_id="c",
                    rule_id="r3",
                    rule_name="R3",
                    severity=ContractSeverity.ERROR,
                    file_path=Path("b.py"),
                ),
            ]
        )
        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.has_blocking_violations is True

    def test_only_warnings_not_blocking(self) -> None:
        report = EvaluationReport(
            violations=[
                Violation(
                    contract_id="c",
                    rule_id="r1",
                    rule_name="R1",
                    severity=ContractSeverity.WARNING,
                    file_path=Path("a.py"),
                ),
            ]
        )
        assert report.has_blocking_violations is False
        assert report.warning_count == 1

    def test_info_not_counted_as_warning_or_error(self) -> None:
        report = EvaluationReport(
            violations=[
                Violation(
                    contract_id="c",
                    rule_id="r1",
                    rule_name="R1",
                    severity=ContractSeverity.INFO,
                    file_path=Path("a.py"),
                ),
            ]
        )
        assert report.error_count == 0
        assert report.warning_count == 0
        assert report.has_blocking_violations is False

    def test_to_context_string_no_violations(self) -> None:
        report = EvaluationReport()
        assert report.to_context_string() == "No contract violations found."

    def test_to_context_string_with_violations(self) -> None:
        report = EvaluationReport(
            violations=[
                Violation(
                    contract_id="c",
                    rule_id="r1",
                    rule_name="Max Lines",
                    severity=ContractSeverity.ERROR,
                    file_path=Path("src/big.py"),
                    line=100,
                    message="File too long",
                    fix_suggestion="Split it",
                ),
            ]
        )
        text = report.to_context_string()
        assert "1 errors" in text
        assert "big.py" in text
        assert "[ERROR]" in text
        assert "Max Lines:100" in text
        assert "Fix: Split it" in text

    def test_to_context_string_with_conflicts(self) -> None:
        report = EvaluationReport(
            violations=[
                Violation(
                    contract_id="c",
                    rule_id="r1",
                    rule_name="R",
                    severity=ContractSeverity.WARNING,
                    file_path=Path("a.py"),
                ),
            ],
            conflicts=["max_lines conflict: A=100, B=200"],
        )
        text = report.to_context_string()
        assert "Contract Conflicts" in text
        assert "max_lines conflict" in text


# ============================================================================
# _detect_conflicts
# ============================================================================


class TestDetectConflicts:
    """Covers conflicting thresholds across contracts."""

    def test_no_threshold_rules_no_conflicts(self) -> None:
        evaluator = _make_evaluator()
        contract = _make_contract(
            rules=[_make_rule(kind=RuleKind.PATTERN, pattern=r"TODO")]
        )
        conflicts = evaluator._detect_conflicts([contract], [])
        assert conflicts == []

    def test_same_metric_same_value_no_conflict(self) -> None:
        evaluator = _make_evaluator()
        c1 = _make_contract(
            contract_id="c1",
            rules=[
                _make_rule(
                    id="r1",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_lines",
                    threshold_value=500,
                )
            ],
        )
        c2 = _make_contract(
            contract_id="c2",
            rules=[
                _make_rule(
                    id="r2",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_lines",
                    threshold_value=500,
                )
            ],
        )
        conflicts = evaluator._detect_conflicts([c1, c2], [])
        assert conflicts == []

    def test_same_metric_different_values_detected(self) -> None:
        evaluator = _make_evaluator()
        c1 = _make_contract(
            contract_id="c1",
            priority=500,
            rules=[
                _make_rule(
                    id="r1",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_lines",
                    threshold_value=300,
                )
            ],
        )
        c2 = _make_contract(
            contract_id="c2",
            priority=200,
            rules=[
                _make_rule(
                    id="r2",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_lines",
                    threshold_value=500,
                )
            ],
        )
        conflicts = evaluator._detect_conflicts([c1, c2], [])
        assert len(conflicts) == 1
        assert "max_lines" in conflicts[0]
        assert "c1" in conflicts[0]
        assert "c2" in conflicts[0]

    def test_different_metrics_no_conflict(self) -> None:
        evaluator = _make_evaluator()
        c1 = _make_contract(
            contract_id="c1",
            rules=[
                _make_rule(
                    id="r1",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_lines",
                    threshold_value=300,
                )
            ],
        )
        c2 = _make_contract(
            contract_id="c2",
            rules=[
                _make_rule(
                    id="r2",
                    kind=RuleKind.THRESHOLD,
                    threshold_metric="max_function_lines",
                    threshold_value=50,
                )
            ],
        )
        conflicts = evaluator._detect_conflicts([c1, c2], [])
        assert conflicts == []

    def test_three_contracts_conflicting_threshold(self) -> None:
        evaluator = _make_evaluator()
        contracts = [
            _make_contract(
                contract_id=f"c{i}",
                priority=i * 100,
                rules=[
                    _make_rule(
                        id=f"r{i}",
                        kind=RuleKind.THRESHOLD,
                        threshold_metric="max_lines",
                        threshold_value=value,
                    )
                ],
            )
            for i, value in enumerate([100, 200, 100], start=1)
        ]
        conflicts = evaluator._detect_conflicts(contracts, [])
        assert len(conflicts) == 1
        assert "Highest-priority contract wins" in conflicts[0]


# ============================================================================
# Invalid regex handling
# ============================================================================


class TestInvalidRegex:
    """Invalid regex in a rule should skip the rule, not crash."""

    def test_invalid_pattern_skipped(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"[invalid((",  # broken regex
        )
        content = "some code\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert violations == []

    def test_invalid_anti_pattern_skipped(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=None,
            anti_pattern=r"[broken((",
        )
        content = "some code\n"
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "foo.py", content
        )
        assert violations == []

    def test_compiled_pattern_cached(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"TODO")
        content = "TODO: fix\n"
        evaluator._evaluate_pattern_rule(rule, "ctr", tmp_path / "a.py", content)
        # Second call should use cached compiled pattern
        evaluator._evaluate_pattern_rule(rule, "ctr", tmp_path / "b.py", content)
        assert rule.id in evaluator._compiled_patterns

    def test_invalid_regex_cached_as_none(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        rule = _make_rule(kind=RuleKind.PATTERN, pattern=r"[bad((")
        content = "code\n"
        evaluator._evaluate_pattern_rule(rule, "ctr", tmp_path / "a.py", content)
        assert evaluator._compiled_patterns.get(rule.id) is None
        # Second invocation still does not crash
        violations = evaluator._evaluate_pattern_rule(
            rule, "ctr", tmp_path / "b.py", content
        )
        assert violations == []


# ============================================================================
# evaluate_files integration (async)
# ============================================================================


class TestEvaluateFiles:
    """Integration-style tests for the full evaluate_files flow."""

    async def test_file_not_found_adds_error(self, tmp_path: Path) -> None:
        evaluator = _make_evaluator()
        report = await evaluator.evaluate_files([tmp_path / "nonexistent.py"])
        assert len(report.errors) == 1
        assert "not found" in report.errors[0].lower()
        assert report.files_checked == 0

    async def test_evaluates_pattern_rule_on_real_file(self, tmp_path: Path) -> None:
        source = tmp_path / "code.py"
        source.write_text("result = eval('1+1')\n", encoding="utf-8")

        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"eval\(",
            severity=ContractSeverity.ERROR,
        )
        contract = _make_contract(rules=[rule])
        evaluator = _make_evaluator(contracts=[contract])

        report = await evaluator.evaluate_files([source])
        assert report.files_checked == 1
        assert report.has_blocking_violations is True
        assert any("eval(" in (v.matched_text or "") for v in report.violations)

    async def test_strict_mode_escalates_warnings(self, tmp_path: Path) -> None:
        source = tmp_path / "code.py"
        source.write_text("x = TODO\n", encoding="utf-8")

        rule = _make_rule(
            kind=RuleKind.PATTERN,
            pattern=r"TODO",
            severity=ContractSeverity.WARNING,
        )
        contract = _make_contract(rules=[rule])
        evaluator = _make_evaluator(contracts=[contract])

        report = await evaluator.evaluate_files([source], strict_mode=True)
        assert all(v.severity == ContractSeverity.ERROR for v in report.violations)
