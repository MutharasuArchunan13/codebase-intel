"""Contract evaluator — checks code against quality contracts.

This is where contracts meet reality. The evaluator takes a file (or set of files)
and checks them against all applicable contracts, producing a report of violations.

Two modes:
1. Pre-generation: agent reads contracts BEFORE writing code (guidance mode)
2. Post-generation: code is checked AFTER writing (validation mode)

Edge cases:
- Rule regex doesn't compile: skip that rule, log error
- Rule matches in comments/strings (false positive): basic heuristic to
  exclude matches inside comments and string literals
- Threshold metric not computable: some metrics need AST data (complexity),
  others just need line count. Degrade gracefully if AST unavailable.
- File is too large for detailed analysis: report size threshold violation
  but skip per-function checks
- Contract applies but file is unparseable (syntax errors): report what
  we can (line count, import checks) without AST-dependent rules
- Multiple contracts have conflicting rules: detect and report conflicts
  separately from violations
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codebase_intel.contracts.models import (
    ContractRule,
    QualityContract,
    RuleKind,
)
from codebase_intel.core.exceptions import ContractViolationError, ErrorContext
from codebase_intel.core.types import ContractSeverity, GraphNode, Language, NodeKind

if TYPE_CHECKING:
    from codebase_intel.contracts.registry import ContractRegistry
    from codebase_intel.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """A single contract violation found in the code."""

    contract_id: str
    rule_id: str
    rule_name: str
    severity: ContractSeverity
    file_path: Path
    line: int | None = None
    message: str = ""
    fix_suggestion: str | None = None
    matched_text: str | None = None  # The violating code fragment


@dataclass
class EvaluationReport:
    """Complete evaluation report for a set of files."""

    files_checked: int = 0
    rules_evaluated: int = 0
    violations: list[Violation] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    skipped_rules: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == ContractSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == ContractSeverity.WARNING)

    @property
    def has_blocking_violations(self) -> bool:
        """True if any ERROR-level violations exist."""
        return self.error_count > 0

    def to_context_string(self) -> str:
        """Serialize for inclusion in agent context."""
        if not self.violations:
            return "No contract violations found."

        lines = [
            f"## Contract Evaluation: {self.error_count} errors, {self.warning_count} warnings",
            "",
        ]

        # Group by file
        by_file: dict[Path, list[Violation]] = {}
        for v in self.violations:
            by_file.setdefault(v.file_path, []).append(v)

        for fp, violations in by_file.items():
            lines.append(f"### {fp.name}")
            for v in violations:
                badge = "ERROR" if v.severity == ContractSeverity.ERROR else "WARN"
                loc = f":{v.line}" if v.line else ""
                lines.append(f"- [{badge}] {v.rule_name}{loc}: {v.message}")
                if v.fix_suggestion:
                    lines.append(f"  Fix: {v.fix_suggestion}")
            lines.append("")

        if self.conflicts:
            lines.append("### Contract Conflicts")
            for c in self.conflicts:
                lines.append(f"- {c}")

        return "\n".join(lines)


class ContractEvaluator:
    """Evaluates code against quality contracts."""

    def __init__(
        self,
        registry: ContractRegistry,
        storage: GraphStorage | None = None,
    ) -> None:
        self._registry = registry
        self._storage = storage
        self._compiled_patterns: dict[str, re.Pattern[str] | None] = {}

    async def evaluate_files(
        self,
        file_paths: list[Path],
        *,
        strict_mode: bool = False,
    ) -> EvaluationReport:
        """Evaluate a set of files against all applicable contracts.

        Edge cases:
        - File doesn't exist (deleted since task started): skip with warning
        - File is binary: skip
        - File has no applicable contracts: still count as checked (0 violations)
        """
        report = EvaluationReport()
        contracts = self._registry.get_all()

        for fp in file_paths:
            if not fp.exists():
                report.errors.append(f"File not found: {fp}")
                continue

            try:
                content = fp.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as exc:
                report.errors.append(f"Cannot read {fp}: {exc}")
                continue

            report.files_checked += 1
            language = self._detect_language(fp)
            is_test = self._is_test_file(fp)

            # Get graph nodes for this file (if graph available)
            file_nodes: list[GraphNode] = []
            if self._storage:
                file_nodes = await self._storage.get_nodes_by_file(fp)

            for contract in contracts:
                rules = contract.rules_for_file(
                    fp, language=language, is_test=is_test
                )
                for rule in rules:
                    report.rules_evaluated += 1
                    violations = self._evaluate_rule(
                        rule, contract.id, fp, content, file_nodes
                    )
                    if strict_mode:
                        for v in violations:
                            if v.severity == ContractSeverity.WARNING:
                                v.severity = ContractSeverity.ERROR
                    report.violations.extend(violations)

        # Check for contract conflicts
        report.conflicts = self._detect_conflicts(contracts, file_paths)

        return report

    def evaluate_for_guidance(
        self,
        file_paths: list[Path],
    ) -> list[QualityContract]:
        """Get applicable contracts for pre-generation guidance.

        Returns contracts (not violations) so the agent knows the rules
        BEFORE writing code.

        Edge case: file doesn't exist yet (agent will create it). Use the
        intended path to match contracts by scope pattern.
        """
        contracts = self._registry.get_all()
        applicable: list[QualityContract] = []

        for contract in contracts:
            for fp in file_paths:
                language = self._detect_language(fp)
                if contract.scope.matches(fp, language):
                    applicable.append(contract)
                    break

        # Sort by priority (highest first)
        return sorted(applicable, key=lambda c: c.priority, reverse=True)

    def _evaluate_rule(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Evaluate a single rule against a file.

        Dispatches to type-specific evaluation methods.
        """
        if rule.kind == RuleKind.PATTERN:
            return self._evaluate_pattern_rule(rule, contract_id, file_path, content)
        elif rule.kind == RuleKind.THRESHOLD:
            return self._evaluate_threshold_rule(
                rule, contract_id, file_path, content, file_nodes
            )
        elif rule.kind == RuleKind.ARCHITECTURAL:
            return self._evaluate_architectural_rule(
                rule, contract_id, file_path, content, file_nodes
            )
        elif rule.kind == RuleKind.AI_ANTIPATTERN:
            return self._evaluate_ai_antipattern(
                rule, contract_id, file_path, content, file_nodes
            )
        return []

    def _evaluate_pattern_rule(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
    ) -> list[Violation]:
        """Evaluate a regex-based pattern rule.

        Edge cases:
        - Regex matches inside comments: basic heuristic to skip comment lines
        - Regex matches inside strings: much harder to detect, accept some false positives
        - Pattern is invalid regex: compile once, cache, skip if invalid
        - Multi-line patterns: use re.MULTILINE flag
        """
        violations: list[Violation] = []

        if rule.pattern:
            compiled = self._compile_pattern(rule.id, rule.pattern)
            if compiled:
                for line_no, line in enumerate(content.split("\n"), 1):
                    # Basic comment filtering
                    stripped = line.strip()
                    if stripped.startswith(("#", "//", "/*", "*")):
                        continue

                    match = compiled.search(line)
                    if match:
                        violations.append(Violation(
                            contract_id=contract_id,
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.effective_severity,
                            file_path=file_path,
                            line=line_no,
                            message=rule.description,
                            fix_suggestion=rule.fix_suggestion,
                            matched_text=match.group(0),
                        ))

        if rule.anti_pattern:
            # Check that the pattern IS present (absence = violation)
            compiled = self._compile_pattern(f"{rule.id}_anti", rule.anti_pattern)
            if compiled and not compiled.search(content):
                violations.append(Violation(
                    contract_id=contract_id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.effective_severity,
                    file_path=file_path,
                    message=f"Required pattern not found: {rule.anti_pattern}",
                    fix_suggestion=rule.fix_suggestion,
                ))

        return violations

    def _evaluate_threshold_rule(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Evaluate a measurable threshold rule.

        Supported metrics:
        - max_lines: total file line count
        - max_function_lines: per-function line count (requires graph nodes)
        - max_complexity: cyclomatic complexity (future — requires AST analysis)

        Edge case: metric requires graph data but graph is unavailable →
        skip rule with a note in skipped_rules.
        """
        violations: list[Violation] = []
        metric = rule.threshold_metric
        threshold = rule.threshold_value

        if metric is None or threshold is None:
            return []

        if metric == "max_lines":
            line_count = content.count("\n") + 1
            if line_count > threshold:
                violations.append(Violation(
                    contract_id=contract_id,
                    rule_id=rule.id,
                    rule_name=rule.name,
                    severity=rule.effective_severity,
                    file_path=file_path,
                    message=f"File has {line_count} lines (max: {int(threshold)})",
                    fix_suggestion=rule.fix_suggestion,
                ))

        elif metric == "max_function_lines":
            # Check each function/method node
            for node in file_nodes:
                if node.kind in (NodeKind.FUNCTION, NodeKind.METHOD) and node.line_range:
                    func_lines = node.line_range.span
                    if func_lines > threshold:
                        violations.append(Violation(
                            contract_id=contract_id,
                            rule_id=rule.id,
                            rule_name=rule.name,
                            severity=rule.effective_severity,
                            file_path=file_path,
                            line=node.line_range.start,
                            message=(
                                f"Function '{node.name}' has {func_lines} lines "
                                f"(max: {int(threshold)})"
                            ),
                            fix_suggestion=rule.fix_suggestion,
                        ))

        return violations

    def _evaluate_architectural_rule(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Evaluate an architectural rule (structural constraints).

        These rules typically need graph data to check dependency direction,
        layer violations, etc. Without a graph, we can still do basic
        import-based checks.
        """
        # Architectural rules are complex — for now, basic pattern matching
        # on imports. Full evaluation requires graph traversal (future).
        violations: list[Violation] = []

        if rule.pattern:
            return self._evaluate_pattern_rule(rule, contract_id, file_path, content)

        return violations

    def _evaluate_ai_antipattern(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Evaluate AI-specific anti-pattern rules.

        These are heuristic-based and accept some false positives in exchange
        for catching common AI mistakes.

        Edge cases:
        - "No restating comments" might flag legitimate explanatory comments
          → severity is INFO, not ERROR, so it's guidance not blocking
        - "No hallucinated imports" needs the graph to verify → degrade
          to pattern check without graph
        """
        violations: list[Violation] = []

        # Delegate to pattern matching if a pattern is defined
        if rule.pattern:
            return self._evaluate_pattern_rule(rule, contract_id, file_path, content)

        # Rule-specific heuristics
        if rule.id == "no-hallucinated-imports" and self._storage:
            violations.extend(
                self._check_hallucinated_imports(
                    rule, contract_id, file_path, content, file_nodes
                )
            )

        if rule.id == "no-over-abstraction":
            violations.extend(
                self._check_over_abstraction(
                    rule, contract_id, file_path, content, file_nodes
                )
            )

        return violations

    def _check_hallucinated_imports(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Check for imports that don't resolve to real files or packages.

        Edge cases:
        - Standard library imports: always valid (we maintain a known-stdlib list)
        - Third-party imports: check against installed packages (best effort)
        - Relative imports: resolve against file position
        - Dynamic imports: can't check statically, skip
        """
        # This is a placeholder — full implementation would use the graph
        # to verify that import edges resolve to actual nodes
        return []

    def _check_over_abstraction(
        self,
        rule: ContractRule,
        contract_id: str,
        file_path: Path,
        content: str,
        file_nodes: list[GraphNode],
    ) -> list[Violation]:
        """Detect potential over-abstraction patterns.

        Heuristics:
        - Base class with only one subclass
        - Factory function that returns only one type
        - Utility function called from only one place
        - Interface with only one implementation

        This requires graph data to check usage counts.
        """
        # Placeholder — requires graph traversal for caller count analysis
        return []

    def _detect_conflicts(
        self,
        contracts: list[QualityContract],
        file_paths: list[Path],
    ) -> list[str]:
        """Detect rules from different contracts that conflict.

        Edge case: two contracts both have a "max function lines" rule
        but with different thresholds. The stricter one wins.

        Edge case: "no helper functions" + "max 50 lines per function"
        → inherently conflicting for complex logic. Report but don't resolve.
        """
        conflicts: list[str] = []

        # Check for duplicate threshold rules with different values
        threshold_rules: dict[str, list[tuple[str, float, int]]] = {}
        for contract in contracts:
            for rule in contract.rules:
                if rule.kind == RuleKind.THRESHOLD and rule.threshold_metric:
                    threshold_rules.setdefault(rule.threshold_metric, []).append(
                        (contract.id, rule.threshold_value or 0, contract.priority)
                    )

        for metric, rules in threshold_rules.items():
            if len(rules) > 1:
                values = set(r[1] for r in rules)
                if len(values) > 1:
                    details = ", ".join(
                        f"{r[0]}: {r[1]}" for r in sorted(rules, key=lambda x: -x[2])
                    )
                    conflicts.append(
                        f"Conflicting {metric} thresholds: {details}. "
                        f"Highest-priority contract wins."
                    )

        return conflicts

    def _compile_pattern(self, cache_key: str, pattern: str) -> re.Pattern[str] | None:
        """Compile and cache a regex pattern.

        Edge case: invalid regex → log error, return None, skip this rule.
        """
        if cache_key in self._compiled_patterns:
            return self._compiled_patterns[cache_key]

        try:
            compiled = re.compile(pattern, re.MULTILINE)
            self._compiled_patterns[cache_key] = compiled
            return compiled
        except re.error as exc:
            logger.error("Invalid regex in rule %s: %s — %s", cache_key, pattern, exc)
            self._compiled_patterns[cache_key] = None
            return None

    def _detect_language(self, file_path: Path) -> Language:
        """Detect language from file extension."""
        from codebase_intel.graph.parser import detect_language
        return detect_language(file_path)

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file."""
        name = file_path.stem.lower()
        parts = [p.lower() for p in file_path.parts]
        return (
            name.startswith("test_")
            or name.endswith("_test")
            or name.endswith("_spec")
            or any(p in ("tests", "test", "__tests__", "spec") for p in parts)
        )
