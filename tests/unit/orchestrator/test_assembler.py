"""Unit tests for codebase_intel.orchestrator.assembler."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from codebase_intel.core.config import OrchestratorConfig
from codebase_intel.core.types import (
    AssembledContext,
    ContextItem,
    ContextPriority,
)
from codebase_intel.orchestrator.assembler import (
    ContextAssembler,
    estimate_tokens,
)

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_config(**overrides: object) -> OrchestratorConfig:
    defaults: dict[str, object] = {
        "default_budget_tokens": 8000,
        "min_useful_tokens": 500,
        "max_assembly_time_ms": 5000,
        "include_stale_context": True,
        "freshness_decay_days": 30,
    }
    defaults.update(overrides)
    return OrchestratorConfig(**defaults)  # type: ignore[arg-type]


def _make_item(
    content: str = "sample content",
    priority: ContextPriority = ContextPriority.MEDIUM,
    source: str = "graph",
    item_type: str = "file_content",
    freshness_score: float = 1.0,
    tokens: int | None = None,
    metadata: dict | None = None,
) -> ContextItem:
    return ContextItem(
        source=source,
        item_type=item_type,
        priority=priority,
        estimated_tokens=tokens if tokens is not None else estimate_tokens(content),
        content=content,
        metadata=metadata or {},
        freshness_score=freshness_score,
    )


def _make_assembler(
    config: OrchestratorConfig | None = None,
    graph_engine: object | None = None,
    decision_store: object | None = None,
    contract_registry: object | None = None,
    contract_evaluator: object | None = None,
) -> ContextAssembler:
    return ContextAssembler(
        config=config or _make_config(),
        graph_engine=graph_engine,
        decision_store=decision_store,
        contract_registry=contract_registry,
        contract_evaluator=contract_evaluator,
    )


# ============================================================================
# estimate_tokens
# ============================================================================


class TestEstimateTokens:
    """Covers empty string, short string, long string."""

    def test_empty_string_returns_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_short_string_positive_count(self) -> None:
        result = estimate_tokens("hello world")
        assert result > 0
        assert result < 10  # "hello world" should be ~2-3 tokens

    def test_long_string_proportional(self) -> None:
        short = estimate_tokens("word")
        long_text = "word " * 100
        long_count = estimate_tokens(long_text)
        assert long_count > short
        assert long_count > 50  # 100 repetitions should produce many tokens

    def test_none_like_empty(self) -> None:
        """Empty string explicitly returns 0."""
        assert estimate_tokens("") == 0

    def test_unicode_text(self) -> None:
        result = estimate_tokens("hello")
        assert result > 0


# ============================================================================
# ContextAssembler._fit_to_budget
# ============================================================================


class TestFitToBudget:
    """Covers all items fit, some dropped, critical item exceeds budget."""

    def test_all_items_fit_within_budget(self) -> None:
        assembler = _make_assembler()
        items = [
            _make_item(content="short", tokens=10, priority=ContextPriority.HIGH),
            _make_item(content="also short", tokens=15, priority=ContextPriority.MEDIUM),
        ]
        fitted, dropped = assembler._fit_to_budget(items, budget_tokens=100)
        assert len(fitted) == 2
        assert dropped == 0

    def test_some_items_dropped_when_budget_tight(self) -> None:
        assembler = _make_assembler()
        items = [
            _make_item(content="important", tokens=50, priority=ContextPriority.HIGH),
            _make_item(content="extra", tokens=60, priority=ContextPriority.LOW),
        ]
        fitted, dropped = assembler._fit_to_budget(items, budget_tokens=80)
        assert len(fitted) == 1
        assert dropped == 1
        assert fitted[0].content == "important"

    def test_critical_item_exceeds_budget_gets_truncated(self) -> None:
        assembler = _make_assembler()
        long_content = "\n".join(f"line {i}: some content here" for i in range(200))
        items = [
            _make_item(
                content=long_content,
                tokens=estimate_tokens(long_content),
                priority=ContextPriority.CRITICAL,
            ),
        ]
        budget = 50  # Much less than the full content
        fitted, _dropped = assembler._fit_to_budget(items, budget_tokens=budget)
        assert len(fitted) == 1
        assert fitted[0].estimated_tokens <= budget
        is_marked = fitted[0].metadata.get("truncated") is True
        assert is_marked or "truncated" in fitted[0].content.lower()

    def test_zero_budget_returns_empty(self) -> None:
        assembler = _make_assembler()
        items = [_make_item(content="data", tokens=10)]
        fitted, dropped = assembler._fit_to_budget(items, budget_tokens=0)
        assert fitted == []
        assert dropped == 1

    def test_empty_items_returns_empty(self) -> None:
        assembler = _make_assembler()
        fitted, dropped = assembler._fit_to_budget([], budget_tokens=1000)
        assert fitted == []
        assert dropped == 0

    def test_non_critical_item_exceeding_budget_dropped(self) -> None:
        assembler = _make_assembler()
        items = [
            _make_item(content="huge", tokens=500, priority=ContextPriority.HIGH),
        ]
        fitted, dropped = assembler._fit_to_budget(items, budget_tokens=100)
        assert fitted == []
        assert dropped == 1

    def test_critical_truncation_only_on_first_item(self) -> None:
        """Critical truncation only happens when fitted is empty."""
        assembler = _make_assembler()
        items = [
            _make_item(content="small", tokens=10, priority=ContextPriority.HIGH),
            _make_item(
                content="big critical",
                tokens=500,
                priority=ContextPriority.CRITICAL,
            ),
        ]
        fitted, dropped = assembler._fit_to_budget(items, budget_tokens=50)
        # First item fits; second critical item is dropped because fitted is not empty
        assert len(fitted) == 1
        assert dropped == 1


# ============================================================================
# ContextAssembler._truncate_to_fit
# ============================================================================


class TestTruncateToFit:
    """Covers content truncation with marker."""

    def test_item_within_budget_returned_unchanged(self) -> None:
        assembler = _make_assembler()
        item = _make_item(content="short text", tokens=5)
        result = assembler._truncate_to_fit(item, budget_tokens=100)
        assert result.content == "short text"
        assert "truncated" not in result.metadata

    def test_large_item_truncated_with_marker(self) -> None:
        assembler = _make_assembler()
        lines = [f"line {i}: some content that takes tokens" for i in range(100)]
        content = "\n".join(lines)
        item = _make_item(content=content, tokens=estimate_tokens(content))
        result = assembler._truncate_to_fit(item, budget_tokens=50)
        assert result.estimated_tokens <= 50
        assert "truncated to fit token budget" in result.content
        assert result.metadata.get("truncated") is True

    def test_even_first_line_too_large(self) -> None:
        """When even a single line exceeds the budget, return metadata-only stub."""
        assembler = _make_assembler()
        content = "a " * 500  # long single line
        item = _make_item(
            content=content,
            tokens=estimate_tokens(content),
            metadata={"file_path": "huge.py"},
        )
        result = assembler._truncate_to_fit(item, budget_tokens=5)
        assert "truncated" in result.content.lower()
        assert "huge.py" in result.content

    def test_preserves_source_and_priority(self) -> None:
        assembler = _make_assembler()
        content = "\n".join(f"line {i}" for i in range(100))
        item = _make_item(
            content=content,
            tokens=estimate_tokens(content),
            priority=ContextPriority.CRITICAL,
            source="graph",
        )
        result = assembler._truncate_to_fit(item, budget_tokens=30)
        assert result.source == "graph"
        assert result.priority == ContextPriority.CRITICAL


# ============================================================================
# ContextAssembler._priority_sort_key
# ============================================================================


class TestPrioritySortKey:
    """Covers ordering by priority, then freshness, then size."""

    def test_higher_priority_comes_first(self) -> None:
        assembler = _make_assembler()
        critical = _make_item(priority=ContextPriority.CRITICAL, tokens=100)
        low = _make_item(priority=ContextPriority.LOW, tokens=10)
        items = [low, critical]
        items.sort(key=lambda i: assembler._priority_sort_key(i))
        assert items[0].priority == ContextPriority.CRITICAL
        assert items[1].priority == ContextPriority.LOW

    def test_same_priority_fresher_first(self) -> None:
        assembler = _make_assembler()
        stale = _make_item(priority=ContextPriority.HIGH, freshness_score=0.3, tokens=10)
        fresh = _make_item(priority=ContextPriority.HIGH, freshness_score=1.0, tokens=10)
        items = [stale, fresh]
        items.sort(key=lambda i: assembler._priority_sort_key(i))
        assert items[0].freshness_score == 1.0
        assert items[1].freshness_score == 0.3

    def test_same_priority_same_freshness_smaller_first(self) -> None:
        assembler = _make_assembler()
        big = _make_item(priority=ContextPriority.MEDIUM, tokens=500, freshness_score=1.0)
        small = _make_item(priority=ContextPriority.MEDIUM, tokens=10, freshness_score=1.0)
        items = [big, small]
        items.sort(key=lambda i: assembler._priority_sort_key(i))
        assert items[0].estimated_tokens == 10
        assert items[1].estimated_tokens == 500

    def test_full_ordering(self) -> None:
        assembler = _make_assembler()
        items = [
            _make_item(priority=ContextPriority.LOW, tokens=10, freshness_score=1.0),
            _make_item(priority=ContextPriority.CRITICAL, tokens=100, freshness_score=0.5),
            _make_item(priority=ContextPriority.HIGH, tokens=50, freshness_score=1.0),
            _make_item(priority=ContextPriority.HIGH, tokens=20, freshness_score=0.8),
        ]
        items.sort(key=lambda i: assembler._priority_sort_key(i))
        assert items[0].priority == ContextPriority.CRITICAL
        assert items[1].priority == ContextPriority.HIGH
        assert items[1].freshness_score == 1.0
        assert items[2].priority == ContextPriority.HIGH
        assert items[2].freshness_score == 0.8
        assert items[3].priority == ContextPriority.LOW


# ============================================================================
# AssembledContext with no components
# ============================================================================


class TestAssembledContextNoComponents:
    """When no graph/decisions/contracts are available, warnings are populated."""

    async def test_no_components_all_warnings(self) -> None:
        assembler = _make_assembler(
            graph_engine=None,
            decision_store=None,
            contract_registry=None,
        )
        result = await assembler.assemble(
            task_description="Fix the bug",
            file_paths=[Path("/src/app.py")],
        )
        assert isinstance(result, AssembledContext)
        assert len(result.warnings) > 0
        warning_text = " ".join(result.warnings)
        assert "graph" in warning_text.lower()
        assert "decision" in warning_text.lower()

    async def test_no_file_paths_no_graph_items(self) -> None:
        assembler = _make_assembler()
        result = await assembler.assemble(task_description="Greenfield task")
        assert result.items == []
        assert result.total_tokens == 0

    async def test_partial_init_warning(self) -> None:
        mock_registry = MagicMock()
        mock_registry.get_for_file.return_value = []
        assembler = _make_assembler(
            contract_registry=mock_registry,
            graph_engine=None,
            decision_store=None,
        )
        result = await assembler.assemble(
            task_description="Fix bug",
            file_paths=[Path("/src/main.py")],
        )
        warning_text = " ".join(result.warnings)
        assert "partial" in warning_text.lower() or "missing" in warning_text.lower()

    async def test_assembly_time_tracked(self) -> None:
        assembler = _make_assembler()
        result = await assembler.assemble(task_description="Quick task")
        assert result.assembly_time_ms >= 0


# ============================================================================
# _detect_contradictions
# ============================================================================


class TestDetectContradictions:
    """Covers stale decisions conflicting with active contracts."""

    def test_no_items_no_contradictions(self) -> None:
        assembler = _make_assembler()
        contradictions = assembler._detect_contradictions([])
        assert contradictions == []

    def test_stale_decision_with_contract_flagged(self) -> None:
        assembler = _make_assembler()
        stale_decision = _make_item(
            item_type="decision",
            source="decisions",
            freshness_score=0.2,
            metadata={"decision_id": "DEC-001", "file_path": "src/auth.py"},
        )
        active_contract = _make_item(
            item_type="contract_rule",
            source="contracts",
            freshness_score=1.0,
            metadata={"contract_id": "auth-rules"},
        )
        contradictions = assembler._detect_contradictions(
            [stale_decision, active_contract]
        )
        assert len(contradictions) == 1
        assert "DEC-001" in contradictions[0]
        assert "auth-rules" in contradictions[0]

    def test_fresh_decision_with_contract_no_flag(self) -> None:
        assembler = _make_assembler()
        fresh_decision = _make_item(
            item_type="decision",
            source="decisions",
            freshness_score=0.9,
            metadata={"decision_id": "DEC-002", "file_path": "src/core.py"},
        )
        contract = _make_item(
            item_type="contract_rule",
            source="contracts",
            freshness_score=1.0,
            metadata={"contract_id": "core-rules"},
        )
        contradictions = assembler._detect_contradictions(
            [fresh_decision, contract]
        )
        assert contradictions == []

    def test_stale_decision_without_contract_no_flag(self) -> None:
        assembler = _make_assembler()
        stale_decision = _make_item(
            item_type="decision",
            source="decisions",
            freshness_score=0.2,
            metadata={"decision_id": "DEC-003", "file_path": "src/old.py"},
        )
        contradictions = assembler._detect_contradictions([stale_decision])
        assert contradictions == []

    def test_multiple_stale_decisions_with_contract(self) -> None:
        assembler = _make_assembler()
        items = [
            _make_item(
                item_type="decision",
                freshness_score=0.1,
                metadata={"decision_id": f"DEC-{i}", "file_path": f"src/f{i}.py"},
            )
            for i in range(3)
        ] + [
            _make_item(
                item_type="contract_rule",
                freshness_score=1.0,
                metadata={"contract_id": "rules"},
            ),
        ]
        contradictions = assembler._detect_contradictions(items)
        assert len(contradictions) == 3

    def test_decision_without_file_path_skipped(self) -> None:
        assembler = _make_assembler()
        stale_no_file = _make_item(
            item_type="decision",
            freshness_score=0.1,
            metadata={"decision_id": "DEC-X"},
        )
        contract = _make_item(
            item_type="contract_rule",
            freshness_score=1.0,
            metadata={"contract_id": "rules"},
        )
        contradictions = assembler._detect_contradictions(
            [stale_no_file, contract]
        )
        # decision_file is empty string which is falsy, so no contradiction flagged
        assert contradictions == []

    def test_non_decision_items_ignored(self) -> None:
        assembler = _make_assembler()
        file_item = _make_item(
            item_type="file_content",
            freshness_score=0.1,
            metadata={"file_path": "stale.py"},
        )
        contract = _make_item(
            item_type="contract_rule",
            freshness_score=1.0,
            metadata={"contract_id": "rules"},
        )
        contradictions = assembler._detect_contradictions([file_item, contract])
        assert contradictions == []
