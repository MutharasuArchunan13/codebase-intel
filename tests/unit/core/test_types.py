"""Tests for core type definitions.

Covers all Pydantic models, enums, validators, computed properties,
and edge cases documented in the types module.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from codebase_intel.core.types import (
    AssembledContext,
    CodeAnchor,
    ContextItem,
    ContextPriority,
    ContractSeverity,
    DecisionStatus,
    DriftLevel,
    EdgeKind,
    FileFingerprint,
    GraphEdge,
    GraphNode,
    Language,
    LineRange,
    NodeKind,
    TokenBudget,
)

# Stable test path that avoids S108 /tmp warnings
TEST_DIR = Path("/var/data/project")


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestNodeKind:
    def test_all_values_present(self) -> None:
        expected = {
            "module",
            "class",
            "function",
            "method",
            "variable",
            "interface",
            "type_alias",
            "endpoint",
            "config",
            "unknown",
        }
        assert {kind.value for kind in NodeKind} == expected

    def test_string_coercion(self) -> None:
        assert str(NodeKind.MODULE) == "NodeKind.MODULE"
        assert NodeKind.MODULE.value == "module"

    def test_is_str_subclass(self) -> None:
        assert isinstance(NodeKind.MODULE, str)


class TestEdgeKind:
    def test_all_values_present(self) -> None:
        expected = {
            "imports",
            "dynamic_import",
            "calls",
            "inherits",
            "implements",
            "instantiates",
            "reads",
            "writes",
            "depends_on",
            "tests",
            "configures",
            "re_exports",
        }
        assert {kind.value for kind in EdgeKind} == expected


class TestLanguage:
    def test_all_values_present(self) -> None:
        expected = {
            "python",
            "javascript",
            "typescript",
            "tsx",
            "go",
            "rust",
            "java",
            "ruby",
            "unknown",
        }
        assert {lang.value for lang in Language} == expected


class TestDecisionStatus:
    def test_all_values_present(self) -> None:
        expected = {"draft", "active", "superseded", "deprecated", "expired"}
        assert {s.value for s in DecisionStatus} == expected


class TestContractSeverity:
    def test_all_values_present(self) -> None:
        expected = {"error", "warning", "info"}
        assert {s.value for s in ContractSeverity} == expected


class TestDriftLevel:
    def test_all_values_present(self) -> None:
        expected = {"none", "low", "medium", "high", "critical"}
        assert {d.value for d in DriftLevel} == expected


# ---------------------------------------------------------------------------
# FileFingerprint
# ---------------------------------------------------------------------------


class TestFileFingerprint:
    def test_valid_construction(self) -> None:
        now = datetime.now(tz=UTC)
        fp = FileFingerprint(
            path=TEST_DIR / "test.py",
            content_hash="abc123",
            size_bytes=100,
            last_modified=now,
            language=Language.PYTHON,
        )
        assert fp.content_hash == "abc123"
        assert fp.size_bytes == 100
        assert fp.language == Language.PYTHON

    def test_naive_datetime_gets_utc(self) -> None:
        naive = datetime(2025, 1, 1, 12, 0, 0)
        fp = FileFingerprint(
            path=TEST_DIR / "test.py",
            content_hash="abc",
            size_bytes=0,
            last_modified=naive,
        )
        assert fp.last_modified.tzinfo is UTC

    def test_non_utc_datetime_converted_to_utc(self) -> None:
        eastern = timezone(offset=timedelta(hours=-5))
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=eastern)
        fp = FileFingerprint(
            path=TEST_DIR / "test.py",
            content_hash="abc",
            size_bytes=0,
            last_modified=ts,
        )
        assert fp.last_modified.tzinfo is UTC
        assert fp.last_modified.hour == 17

    def test_path_normalization_resolves_relative(self) -> None:
        fp = FileFingerprint(
            path=Path("relative/path/test.py"),
            content_hash="abc",
            size_bytes=0,
            last_modified=datetime.now(tz=UTC),
        )
        assert fp.path.is_absolute()

    def test_path_normalization_resolves_dotdot(self) -> None:
        fp = FileFingerprint(
            path=TEST_DIR / "a" / ".." / "b" / "test.py",
            content_hash="abc",
            size_bytes=0,
            last_modified=datetime.now(tz=UTC),
        )
        assert ".." not in str(fp.path)
        assert str(fp.path).endswith("b/test.py")

    def test_size_bytes_rejects_negative(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            FileFingerprint(
                path=TEST_DIR / "test.py",
                content_hash="abc",
                size_bytes=-1,
                last_modified=datetime.now(tz=UTC),
            )

    def test_empty_file_valid(self) -> None:
        fp = FileFingerprint(
            path=TEST_DIR / "empty.py",
            content_hash="d41d8cd98f00b204e9800998ecf8427e",
            size_bytes=0,
            last_modified=datetime.now(tz=UTC),
        )
        assert fp.size_bytes == 0

    def test_default_language_is_unknown(self) -> None:
        fp = FileFingerprint(
            path=TEST_DIR / "test.txt",
            content_hash="abc",
            size_bytes=10,
            last_modified=datetime.now(tz=UTC),
        )
        assert fp.language == Language.UNKNOWN

    def test_frozen_immutability(self) -> None:
        fp = FileFingerprint(
            path=TEST_DIR / "test.py",
            content_hash="abc",
            size_bytes=100,
            last_modified=datetime.now(tz=UTC),
        )
        with pytest.raises(ValidationError, match=r"frozen"):
            fp.size_bytes = 200  # type: ignore[misc]


# ---------------------------------------------------------------------------
# LineRange
# ---------------------------------------------------------------------------


class TestLineRange:
    def test_valid_range(self) -> None:
        lr = LineRange(start=1, end=10)
        assert lr.start == 1
        assert lr.end == 10

    def test_single_line_range(self) -> None:
        lr = LineRange(start=5, end=5)
        assert lr.start == lr.end
        assert lr.span == 1

    def test_span_calculation(self) -> None:
        lr = LineRange(start=3, end=7)
        assert lr.span == 5

    def test_start_after_end_raises(self) -> None:
        with pytest.raises(ValidationError, match=r"must be <= end"):
            LineRange(start=10, end=5)

    def test_zero_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 1"):
            LineRange(start=0, end=5)

    def test_zero_end_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 1"):
            LineRange(start=1, end=0)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 1"):
            LineRange(start=-1, end=5)

    def test_frozen_immutability(self) -> None:
        lr = LineRange(start=1, end=10)
        with pytest.raises(ValidationError, match=r"frozen"):
            lr.start = 2  # type: ignore[misc]

    def test_large_range_span(self) -> None:
        lr = LineRange(start=1, end=10000)
        assert lr.span == 10000


# ---------------------------------------------------------------------------
# CodeAnchor
# ---------------------------------------------------------------------------


class TestCodeAnchor:
    def test_valid_construction(self) -> None:
        anchor = CodeAnchor(
            file_path=TEST_DIR / "module.py",
            line_range=LineRange(start=10, end=20),
            symbol_name="MyClass.method",
            content_hash="abc123",
        )
        assert anchor.symbol_name == "MyClass.method"
        assert anchor.content_hash == "abc123"

    def test_path_normalization(self) -> None:
        anchor = CodeAnchor(file_path=TEST_DIR / "a" / ".." / "b" / "test.py")
        assert anchor.file_path.is_absolute()
        assert ".." not in str(anchor.file_path)

    def test_optional_fields_default_to_none(self) -> None:
        anchor = CodeAnchor(file_path=TEST_DIR / "test.py")
        assert anchor.line_range is None
        assert anchor.symbol_name is None
        assert anchor.content_hash is None

    def test_is_orphaned_when_file_missing(self) -> None:
        anchor = CodeAnchor(file_path=TEST_DIR / "test.py")
        existing = {(TEST_DIR / "other.py").resolve()}
        assert anchor.is_orphaned(existing) is True

    def test_is_not_orphaned_when_file_exists(self) -> None:
        resolved = (TEST_DIR / "test.py").resolve()
        anchor = CodeAnchor(file_path=TEST_DIR / "test.py")
        existing = {resolved}
        assert anchor.is_orphaned(existing) is False

    def test_is_orphaned_with_empty_set(self) -> None:
        anchor = CodeAnchor(file_path=TEST_DIR / "test.py")
        assert anchor.is_orphaned(set()) is True

    def test_frozen_immutability(self) -> None:
        anchor = CodeAnchor(file_path=TEST_DIR / "test.py")
        with pytest.raises(ValidationError, match=r"frozen"):
            anchor.symbol_name = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TokenBudget
# ---------------------------------------------------------------------------


class TestTokenBudget:
    def test_basic_usable_calculation(self) -> None:
        budget = TokenBudget(total=1000, reserved_for_response=200, safety_margin_pct=0.1)
        # raw = 1000 - 200 = 800, margin = 80, usable = 720
        assert budget.usable == 720

    def test_zero_reserve(self) -> None:
        budget = TokenBudget(total=1000, reserved_for_response=0, safety_margin_pct=0.1)
        # raw = 1000, margin = 100, usable = 900
        assert budget.usable == 900

    def test_zero_safety_margin(self) -> None:
        budget = TokenBudget(total=1000, reserved_for_response=200, safety_margin_pct=0.0)
        # raw = 800, margin = 0, usable = 800
        assert budget.usable == 800

    def test_max_safety_margin(self) -> None:
        budget = TokenBudget(total=1000, reserved_for_response=0, safety_margin_pct=0.5)
        # raw = 1000, margin = 500, usable = 500
        assert budget.usable == 500

    def test_safety_margin_above_max_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"less than or equal to 0\.5"):
            TokenBudget(total=1000, safety_margin_pct=0.6)

    def test_safety_margin_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            TokenBudget(total=1000, safety_margin_pct=-0.1)

    def test_total_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than 0"):
            TokenBudget(total=0)

    def test_negative_total_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than 0"):
            TokenBudget(total=-100)

    def test_reserved_exceeds_total_gives_zero_usable(self) -> None:
        budget = TokenBudget(total=100, reserved_for_response=200, safety_margin_pct=0.0)
        # raw = -100, but max(0, ...) clamps to 0
        assert budget.usable == 0

    def test_usable_never_negative(self) -> None:
        budget = TokenBudget(total=1, reserved_for_response=0, safety_margin_pct=0.5)
        assert budget.usable >= 0

    def test_default_values(self) -> None:
        budget = TokenBudget(total=1000)
        assert budget.reserved_for_response == 0
        assert budget.safety_margin_pct == pytest.approx(0.1)

    def test_frozen_immutability(self) -> None:
        budget = TokenBudget(total=1000)
        with pytest.raises(ValidationError, match=r"frozen"):
            budget.total = 2000  # type: ignore[misc]

    def test_reserved_negative_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            TokenBudget(total=1000, reserved_for_response=-10)


# ---------------------------------------------------------------------------
# GraphNode
# ---------------------------------------------------------------------------


class TestGraphNode:
    @pytest.fixture()
    def sample_node(self) -> GraphNode:
        return GraphNode(
            node_id="abc123",
            kind=NodeKind.FUNCTION,
            name="my_func",
            qualified_name="module.my_func",
            file_path=TEST_DIR / "module.py",
            language=Language.PYTHON,
        )

    def test_valid_construction(self, sample_node: GraphNode) -> None:
        assert sample_node.node_id == "abc123"
        assert sample_node.kind == NodeKind.FUNCTION
        assert sample_node.name == "my_func"

    def test_default_flags(self, sample_node: GraphNode) -> None:
        assert sample_node.is_generated is False
        assert sample_node.is_external is False
        assert sample_node.is_test is False
        assert sample_node.is_entry_point is False

    def test_default_optional_fields(self, sample_node: GraphNode) -> None:
        assert sample_node.line_range is None
        assert sample_node.content_hash is None
        assert sample_node.docstring is None
        assert sample_node.metadata == {}

    def test_make_id_deterministic(self) -> None:
        path = TEST_DIR / "module.py"
        kind = NodeKind.FUNCTION
        name = "my_func"
        id1 = GraphNode.make_id(path, kind, name)
        id2 = GraphNode.make_id(path, kind, name)
        assert id1 == id2

    def test_make_id_is_16_char_hex(self) -> None:
        node_id = GraphNode.make_id(TEST_DIR / "x.py", NodeKind.CLASS, "Foo")
        assert len(node_id) == 16
        int(node_id, 16)  # Raises ValueError if not valid hex

    def test_make_id_uses_sha256(self) -> None:
        path = TEST_DIR / "module.py"
        kind = NodeKind.FUNCTION
        name = "my_func"
        raw = f"{path.resolve()}:{kind.value}:{name}"
        expected = hashlib.sha256(raw.encode()).hexdigest()[:16]
        assert GraphNode.make_id(path, kind, name) == expected

    def test_make_id_different_for_different_kinds(self) -> None:
        path = TEST_DIR / "module.py"
        name = "same_name"
        id_func = GraphNode.make_id(path, NodeKind.FUNCTION, name)
        id_class = GraphNode.make_id(path, NodeKind.CLASS, name)
        assert id_func != id_class

    def test_make_id_different_for_different_paths(self) -> None:
        kind = NodeKind.FUNCTION
        name = "same_name"
        id1 = GraphNode.make_id(TEST_DIR / "a.py", kind, name)
        id2 = GraphNode.make_id(TEST_DIR / "b.py", kind, name)
        assert id1 != id2

    def test_make_id_different_for_different_names(self) -> None:
        path = TEST_DIR / "module.py"
        kind = NodeKind.FUNCTION
        id1 = GraphNode.make_id(path, kind, "func_a")
        id2 = GraphNode.make_id(path, kind, "func_b")
        assert id1 != id2

    def test_make_id_all_node_kinds(self) -> None:
        """Ensure make_id produces unique IDs for every NodeKind with the same path/name."""
        path = TEST_DIR / "module.py"
        name = "entity"
        ids = {GraphNode.make_id(path, kind, name) for kind in NodeKind}
        assert len(ids) == len(NodeKind)

    def test_frozen_immutability(self, sample_node: GraphNode) -> None:
        with pytest.raises(ValidationError, match=r"frozen"):
            sample_node.name = "changed"  # type: ignore[misc]

    def test_with_line_range(self) -> None:
        node = GraphNode(
            node_id="abc",
            kind=NodeKind.METHOD,
            name="do_stuff",
            qualified_name="Cls.do_stuff",
            file_path=TEST_DIR / "test.py",
            line_range=LineRange(start=5, end=15),
        )
        assert node.line_range is not None
        assert node.line_range.span == 11

    def test_metadata_dict(self) -> None:
        node = GraphNode(
            node_id="abc",
            kind=NodeKind.MODULE,
            name="mod",
            qualified_name="mod",
            file_path=TEST_DIR / "mod.py",
            metadata={"complexity": 5, "loc": 120},
        )
        assert node.metadata["complexity"] == 5
        assert node.metadata["loc"] == 120


# ---------------------------------------------------------------------------
# GraphEdge
# ---------------------------------------------------------------------------


class TestGraphEdge:
    def test_valid_construction(self) -> None:
        edge = GraphEdge(
            source_id="aaa",
            target_id="bbb",
            kind=EdgeKind.IMPORTS,
        )
        assert edge.source_id == "aaa"
        assert edge.target_id == "bbb"
        assert edge.kind == EdgeKind.IMPORTS

    def test_default_confidence_is_one(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS)
        assert edge.confidence == pytest.approx(1.0)

    def test_default_is_type_only_false(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS)
        assert edge.is_type_only is False

    def test_confidence_lower_bound(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.DYNAMIC_IMPORT, confidence=0.0)
        assert edge.confidence == pytest.approx(0.0)

    def test_confidence_upper_bound(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.IMPORTS, confidence=1.0)
        assert edge.confidence == pytest.approx(1.0)

    def test_confidence_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS, confidence=-0.1)

    def test_confidence_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"less than or equal to 1"):
            GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS, confidence=1.1)

    def test_type_only_flag(self) -> None:
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            kind=EdgeKind.IMPORTS,
            is_type_only=True,
        )
        assert edge.is_type_only is True

    def test_frozen_immutability(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.CALLS)
        with pytest.raises(ValidationError, match=r"frozen"):
            edge.confidence = 0.5  # type: ignore[misc]

    def test_default_metadata_empty(self) -> None:
        edge = GraphEdge(source_id="a", target_id="b", kind=EdgeKind.INHERITS)
        assert edge.metadata == {}

    def test_metadata_preserved(self) -> None:
        edge = GraphEdge(
            source_id="a",
            target_id="b",
            kind=EdgeKind.CALLS,
            metadata={"call_count": 3},
        )
        assert edge.metadata["call_count"] == 3

    def test_circular_edge_allowed(self) -> None:
        edge = GraphEdge(source_id="same", target_id="same", kind=EdgeKind.CALLS)
        assert edge.source_id == edge.target_id


# ---------------------------------------------------------------------------
# ContextItem
# ---------------------------------------------------------------------------


class TestContextItem:
    @pytest.fixture()
    def sample_item(self) -> ContextItem:
        return ContextItem(
            source="graph",
            item_type="file_content",
            priority=ContextPriority.HIGH,
            estimated_tokens=100,
            content="def foo(): pass",
        )

    def test_valid_construction(self, sample_item: ContextItem) -> None:
        assert sample_item.source == "graph"
        assert sample_item.item_type == "file_content"
        assert sample_item.priority == ContextPriority.HIGH
        assert sample_item.estimated_tokens == 100

    def test_default_freshness_score(self, sample_item: ContextItem) -> None:
        assert sample_item.freshness_score == pytest.approx(1.0)

    def test_freshness_score_lower_bound(self) -> None:
        item = ContextItem(
            source="decisions",
            item_type="decision",
            priority=ContextPriority.LOW,
            estimated_tokens=50,
            content="old decision",
            freshness_score=0.0,
        )
        assert item.freshness_score == pytest.approx(0.0)

    def test_freshness_score_upper_bound(self) -> None:
        item = ContextItem(
            source="contracts",
            item_type="contract_rule",
            priority=ContextPriority.CRITICAL,
            estimated_tokens=30,
            content="rule",
            freshness_score=1.0,
        )
        assert item.freshness_score == pytest.approx(1.0)

    def test_freshness_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            ContextItem(
                source="graph",
                item_type="file_content",
                priority=ContextPriority.HIGH,
                estimated_tokens=10,
                content="x",
                freshness_score=-0.1,
            )

    def test_freshness_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"less than or equal to 1"):
            ContextItem(
                source="graph",
                item_type="file_content",
                priority=ContextPriority.HIGH,
                estimated_tokens=10,
                content="x",
                freshness_score=1.1,
            )

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValidationError, match=r"greater than or equal to 0"):
            ContextItem(
                source="graph",
                item_type="file_content",
                priority=ContextPriority.HIGH,
                estimated_tokens=-1,
                content="x",
            )

    def test_zero_tokens_allowed(self) -> None:
        item = ContextItem(
            source="graph",
            item_type="warning",
            priority=ContextPriority.LOW,
            estimated_tokens=0,
            content="",
        )
        assert item.estimated_tokens == 0

    def test_default_metadata_empty(self, sample_item: ContextItem) -> None:
        assert sample_item.metadata == {}

    def test_all_priority_levels(self) -> None:
        for priority in ContextPriority:
            item = ContextItem(
                source="graph",
                item_type="file_content",
                priority=priority,
                estimated_tokens=10,
                content="x",
            )
            assert item.priority == priority


# ---------------------------------------------------------------------------
# AssembledContext
# ---------------------------------------------------------------------------


class TestAssembledContext:
    def test_empty_context(self) -> None:
        ctx = AssembledContext()
        assert ctx.items == []
        assert ctx.total_tokens == 0
        assert ctx.budget_tokens == 0
        assert ctx.truncated is False
        assert ctx.dropped_count == 0
        assert ctx.conflicts == []
        assert ctx.warnings == []
        assert ctx.assembly_time_ms == pytest.approx(0.0)

    def test_default_values(self) -> None:
        ctx = AssembledContext()
        assert ctx.truncated is False
        assert ctx.dropped_count == 0

    def test_truncated_context(self) -> None:
        ctx = AssembledContext(
            total_tokens=500,
            budget_tokens=1000,
            truncated=True,
            dropped_count=3,
        )
        assert ctx.truncated is True
        assert ctx.dropped_count == 3

    def test_context_with_items(self) -> None:
        items = [
            ContextItem(
                source="graph",
                item_type="file_content",
                priority=ContextPriority.CRITICAL,
                estimated_tokens=100,
                content="def main(): pass",
            ),
            ContextItem(
                source="decisions",
                item_type="decision",
                priority=ContextPriority.HIGH,
                estimated_tokens=50,
                content="Use repository pattern",
            ),
        ]
        ctx = AssembledContext(
            items=items,
            total_tokens=150,
            budget_tokens=1000,
        )
        assert len(ctx.items) == 2
        assert ctx.total_tokens == 150

    def test_context_with_conflicts(self) -> None:
        ctx = AssembledContext(
            conflicts=["Decision ADR-001 contradicts ADR-003 on caching strategy"],
        )
        assert len(ctx.conflicts) == 1

    def test_context_with_warnings(self) -> None:
        ctx = AssembledContext(
            warnings=["Graph module partially initialized: missing tree-sitter grammar for Go"],
        )
        assert len(ctx.warnings) == 1

    def test_assembly_time_tracked(self) -> None:
        ctx = AssembledContext(assembly_time_ms=42.5)
        assert ctx.assembly_time_ms == pytest.approx(42.5)

    def test_context_is_mutable(self) -> None:
        """AssembledContext is NOT frozen, so mutation should work."""
        ctx = AssembledContext()
        ctx.truncated = True
        ctx.total_tokens = 500
        assert ctx.truncated is True
        assert ctx.total_tokens == 500
