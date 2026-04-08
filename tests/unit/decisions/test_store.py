"""Tests for decision store — YAML persistence and querying.

Covers:
- DecisionStore.load_all (empty dir, single file, multiple files, malformed YAML)
- DecisionStore.save (round-trip save and load)
- DecisionStore.query_by_files (relevance scoring, superseded filtering)
- DecisionStore.find_conflicts (overlapping anchors, supersedes chain)
- DecisionStore.next_id (sequential numbering, gap handling)
- _parse_anchor_shorthand (file:start-end, file:line, file)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from codebase_intel.core.config import DecisionConfig
from codebase_intel.core.types import CodeAnchor, LineRange
from codebase_intel.decisions.models import (
    DecisionRecord,
)
from codebase_intel.decisions.store import DecisionStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def decisions_dir(tmp_path: Path) -> Path:
    """Create and return an empty decisions directory."""
    d = tmp_path / "decisions"
    d.mkdir()
    return d


@pytest.fixture()
def config(decisions_dir: Path) -> DecisionConfig:
    """DecisionConfig pointing to the tmp decisions directory."""
    return DecisionConfig(decisions_dir=decisions_dir)


@pytest.fixture()
def store(config: DecisionConfig, tmp_path: Path) -> DecisionStore:
    """A fresh DecisionStore backed by a temp directory."""
    return DecisionStore(config=config, project_root=tmp_path)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Helper to write a YAML decision file."""
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _decision_data(
    *,
    decision_id: str = "DEC-001",
    title: str = "Use token bucket",
    status: str = "active",
    context: str = "Need rate limiting",
    decision: str = "Token bucket with Redis",
    **extra: Any,
) -> dict[str, Any]:
    """Factory for decision YAML data dicts."""
    base: dict[str, Any] = {
        "id": decision_id,
        "title": title,
        "status": status,
        "context": context,
        "decision": decision,
    }
    base.update(extra)
    return base


def _minimal_record(**overrides: Any) -> DecisionRecord:
    """Factory for a DecisionRecord with only required fields."""
    defaults: dict[str, Any] = {
        "id": "DEC-001",
        "title": "Use token bucket",
        "context": "Need rate limiting",
        "decision": "Token bucket with Redis",
    }
    defaults.update(overrides)
    return DecisionRecord(**defaults)


# ---------------------------------------------------------------------------
# DecisionStore.load_all
# ---------------------------------------------------------------------------


class TestLoadAll:
    async def test_empty_directory(self, store: DecisionStore) -> None:
        records = await store.load_all()
        assert records == []

    async def test_directory_does_not_exist(
        self, tmp_path: Path,
    ) -> None:
        nonexistent = tmp_path / "nonexistent"
        cfg = DecisionConfig(decisions_dir=nonexistent)
        s = DecisionStore(config=cfg, project_root=tmp_path)
        records = await s.load_all()
        assert records == []

    async def test_single_file(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        records = await store.load_all()
        assert len(records) == 1
        assert records[0].id == "DEC-001"

    async def test_multiple_files(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        for i in range(1, 4):
            _write_yaml(
                decisions_dir / f"DEC-{i:03d}.yaml",
                _decision_data(decision_id=f"DEC-{i:03d}", title=f"Decision {i}"),
            )
        records = await store.load_all()
        assert len(records) == 3
        ids = {r.id for r in records}
        assert ids == {"DEC-001", "DEC-002", "DEC-003"}

    async def test_malformed_yaml_skipped(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        # Valid file
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        # Malformed YAML
        (decisions_dir / "DEC-002.yaml").write_text(
            "id: DEC-002\ntitle: Bad\n  broken: yaml: [[[",
            encoding="utf-8",
        )
        records = await store.load_all()
        assert len(records) == 1
        assert records[0].id == "DEC-001"

    async def test_empty_yaml_file_skipped(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        (decisions_dir / "DEC-002.yaml").write_text("", encoding="utf-8")
        records = await store.load_all()
        assert len(records) == 1

    async def test_non_dict_yaml_skipped(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        """YAML file containing a list instead of a mapping should be skipped."""
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        (decisions_dir / "DEC-002.yaml").write_text(
            "- item1\n- item2\n", encoding="utf-8",
        )
        records = await store.load_all()
        assert len(records) == 1

    async def test_only_yaml_files_loaded(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        (decisions_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
        records = await store.load_all()
        assert len(records) == 1

    async def test_duplicate_ids_keeps_latest(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        """When two files share the same decision ID, the later-sorted file wins."""
        _write_yaml(
            decisions_dir / "aaa-DEC-001.yaml",
            _decision_data(title="First"),
        )
        _write_yaml(
            decisions_dir / "zzz-DEC-001.yaml",
            _decision_data(title="Second"),
        )
        records = await store.load_all()
        assert len(records) == 1
        # sorted() processes files alphabetically, so zzz overwrites aaa
        assert records[0].title == "Second"

    async def test_file_with_code_anchor_shorthand(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        data = _decision_data(
            code_anchors=["src/api/handler.py:15-82"],
        )
        _write_yaml(decisions_dir / "DEC-001.yaml", data)
        records = await store.load_all()
        assert len(records) == 1
        assert len(records[0].code_anchors) == 1
        anchor = records[0].code_anchors[0]
        assert anchor.line_range is not None
        assert anchor.line_range.start == 15
        assert anchor.line_range.end == 82

    async def test_file_with_alternatives_and_constraints(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        data = _decision_data(
            alternatives=[{
                "name": "Sliding window",
                "description": "Count in sliding window",
                "rejection_reason": "Higher memory",
            }],
            constraints=[{
                "description": "p99 < 200ms",
                "source": "sla",
                "is_hard": True,
            }],
        )
        _write_yaml(decisions_dir / "DEC-001.yaml", data)
        records = await store.load_all()
        assert len(records[0].alternatives) == 1
        assert records[0].alternatives[0].name == "Sliding window"
        assert len(records[0].constraints) == 1
        assert records[0].constraints[0].description == "p99 < 200ms"

    async def test_constraint_as_string_shorthand(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        data = _decision_data(
            constraints=["Must use HTTPS"],
        )
        _write_yaml(decisions_dir / "DEC-001.yaml", data)
        records = await store.load_all()
        assert len(records[0].constraints) == 1
        assert records[0].constraints[0].description == "Must use HTTPS"
        assert records[0].constraints[0].source == "unknown"


# ---------------------------------------------------------------------------
# DecisionStore.save (round-trip)
# ---------------------------------------------------------------------------


class TestSave:
    async def test_save_creates_file(self, store: DecisionStore) -> None:
        record = _minimal_record()
        path = await store.save(record)
        assert path.exists()
        assert path.name == "DEC-001.yaml"

    async def test_round_trip_preserves_data(
        self, store: DecisionStore,
    ) -> None:
        original = _minimal_record(
            tags=["architecture"],
            consequences=["Lower memory"],
            author="alice",
        )
        await store.save(original)

        # Clear cache to force reload from disk
        store._cache.clear()
        store._cache_mtimes.clear()

        loaded = await store.load_all()
        assert len(loaded) == 1
        reloaded = loaded[0]
        assert reloaded.id == original.id
        assert reloaded.title == original.title
        assert reloaded.context == original.context
        assert reloaded.decision == original.decision
        assert reloaded.tags == original.tags
        assert reloaded.consequences == original.consequences
        assert reloaded.author == original.author

    async def test_save_with_code_anchors_round_trip(
        self, store: DecisionStore, tmp_path: Path,
    ) -> None:
        anchor_file = tmp_path / "service.py"
        anchor_file.touch()
        anchor = CodeAnchor(
            file_path=anchor_file,
            line_range=LineRange(start=10, end=50),
        )
        original = _minimal_record(code_anchors=[anchor])
        await store.save(original)

        store._cache.clear()
        store._cache_mtimes.clear()

        loaded = await store.load_all()
        assert len(loaded) == 1
        assert len(loaded[0].code_anchors) == 1
        loaded_anchor = loaded[0].code_anchors[0]
        assert loaded_anchor.line_range is not None
        assert loaded_anchor.line_range.start == 10
        assert loaded_anchor.line_range.end == 50

    async def test_save_creates_directory_if_missing(
        self, tmp_path: Path,
    ) -> None:
        new_dir = tmp_path / "new" / "decisions"
        cfg = DecisionConfig(decisions_dir=new_dir)
        s = DecisionStore(config=cfg, project_root=tmp_path)

        record = _minimal_record()
        path = await s.save(record)
        assert path.exists()
        assert new_dir.exists()

    async def test_save_overwrites_existing_file(
        self, store: DecisionStore,
    ) -> None:
        original = _minimal_record(title="Original Title")
        await store.save(original)

        # Pydantic models are frozen, so create a new one with the same ID
        updated = _minimal_record(title="Updated Title")
        await store.save(updated)

        store._cache.clear()
        store._cache_mtimes.clear()

        loaded = await store.load_all()
        assert len(loaded) == 1
        assert loaded[0].title == "Updated Title"

    async def test_save_sanitizes_filename(
        self, store: DecisionStore,
    ) -> None:
        record = _minimal_record(id="DEC/special\\chars")
        path = await store.save(record)
        assert "/" not in path.name
        assert "\\" not in path.name
        assert path.exists()


# ---------------------------------------------------------------------------
# DecisionStore.query_by_files
# ---------------------------------------------------------------------------


class TestQueryByFiles:
    async def test_empty_store_returns_empty(
        self, store: DecisionStore, tmp_path: Path,
    ) -> None:
        result = await store.query_by_files({tmp_path / "any.py"})
        assert result == []

    async def test_returns_relevant_decisions_sorted(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        target = tmp_path / "src" / "handler.py"
        target.parent.mkdir(parents=True)
        target.touch()
        sibling = tmp_path / "src" / "router.py"
        sibling.touch()

        # Direct match to target
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                code_anchors=[str(target)],
            ),
        )
        # Anchor to sibling (same directory)
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                title="Sibling decision",
                code_anchors=[str(sibling)],
            ),
        )

        results = await store.query_by_files({target})
        assert len(results) >= 1
        # First result should be the direct match with highest score
        assert results[0][0].id == "DEC-001"
        assert results[0][1] == 1.0

    async def test_filters_superseded_decisions(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                status="superseded",
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                status="active",
            ),
        )
        # No anchors means baseline 0.1 score
        results = await store.query_by_files({Path("/any.py")})
        ids = {r[0].id for r in results}
        assert "DEC-001" not in ids
        assert "DEC-002" in ids

    async def test_filters_expired_decisions(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(decision_id="DEC-001", status="expired"),
        )
        results = await store.query_by_files({Path("/any.py")})
        assert len(results) == 0

    async def test_min_relevance_threshold(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        # Decision with no anchors gets 0.1 baseline
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(),
        )
        # With min_relevance=0.5, baseline 0.1 should be excluded
        results = await store.query_by_files(
            {Path("/any.py")}, min_relevance=0.5,
        )
        assert len(results) == 0

        # With min_relevance=0.1, it should be included
        results = await store.query_by_files(
            {Path("/any.py")}, min_relevance=0.1,
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# DecisionStore.find_conflicts
# ---------------------------------------------------------------------------


class TestFindConflicts:
    async def test_no_conflicts_in_empty_store(
        self, store: DecisionStore,
    ) -> None:
        conflicts = await store.find_conflicts()
        assert conflicts == []

    async def test_supersedes_chain_conflict(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        """Both the superseding and superseded decision are active."""
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(decision_id="DEC-001"),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                supersedes="DEC-001",
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 1
        d_a, d_b, reason = conflicts[0]
        assert "supersedes" in reason
        assert {d_a.id, d_b.id} == {"DEC-001", "DEC-002"}

    async def test_overlapping_anchors_whole_file(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        target = tmp_path / "service.py"
        target.touch()

        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                code_anchors=[str(target)],
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                code_anchors=[str(target)],
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 1
        assert "service.py" in conflicts[0][2]

    async def test_overlapping_line_ranges(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        target = tmp_path / "service.py"
        target.touch()

        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                code_anchors=[f"{target}:10-30"],
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                code_anchors=[f"{target}:25-50"],
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 1
        assert "Overlapping" in conflicts[0][2]

    async def test_non_overlapping_line_ranges_no_conflict(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        target = tmp_path / "service.py"
        target.touch()

        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                code_anchors=[f"{target}:1-10"],
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                code_anchors=[f"{target}:20-30"],
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 0

    async def test_different_files_no_conflict(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.touch()
        file_b.touch()

        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                code_anchors=[str(file_a)],
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                code_anchors=[str(file_b)],
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 0

    async def test_non_active_decisions_excluded_from_conflicts(
        self, store: DecisionStore, decisions_dir: Path, tmp_path: Path,
    ) -> None:
        target = tmp_path / "service.py"
        target.touch()

        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(
                decision_id="DEC-001",
                status="active",
                code_anchors=[str(target)],
            ),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(
                decision_id="DEC-002",
                status="deprecated",
                code_anchors=[str(target)],
            ),
        )

        conflicts = await store.find_conflicts()
        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# DecisionStore.next_id
# ---------------------------------------------------------------------------


class TestNextId:
    async def test_empty_store_returns_001(
        self, store: DecisionStore,
    ) -> None:
        next_id = await store.next_id()
        assert next_id == "DEC-001"

    async def test_sequential_after_existing(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(decision_id="DEC-001"),
        )
        _write_yaml(
            decisions_dir / "DEC-002.yaml",
            _decision_data(decision_id="DEC-002"),
        )
        next_id = await store.next_id()
        assert next_id == "DEC-003"

    async def test_gap_handling_uses_max(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        """IDs 001 and 005 exist (gap at 002-004); next should be 006."""
        _write_yaml(
            decisions_dir / "DEC-001.yaml",
            _decision_data(decision_id="DEC-001"),
        )
        _write_yaml(
            decisions_dir / "DEC-005.yaml",
            _decision_data(decision_id="DEC-005"),
        )
        next_id = await store.next_id()
        assert next_id == "DEC-006"

    async def test_non_dec_ids_ignored(
        self, store: DecisionStore, decisions_dir: Path,
    ) -> None:
        """Decisions with non-standard IDs should not affect numbering."""
        _write_yaml(
            decisions_dir / "DEC-003.yaml",
            _decision_data(decision_id="DEC-003"),
        )
        _write_yaml(
            decisions_dir / "CUSTOM-001.yaml",
            _decision_data(decision_id="CUSTOM-001"),
        )
        next_id = await store.next_id()
        assert next_id == "DEC-004"

    async def test_zero_padded_format(
        self, store: DecisionStore,
    ) -> None:
        next_id = await store.next_id()
        assert next_id == "DEC-001"
        assert len(next_id.split("-")[1]) == 3


# ---------------------------------------------------------------------------
# _parse_anchor_shorthand
# ---------------------------------------------------------------------------


class TestParseAnchorShorthand:
    def test_file_with_line_range(self, store: DecisionStore) -> None:
        anchor = store._parse_anchor_shorthand("src/handler.py:15-82")
        assert str(anchor.file_path).endswith("handler.py")
        assert anchor.line_range is not None
        assert anchor.line_range.start == 15
        assert anchor.line_range.end == 82

    def test_file_with_single_line(self, store: DecisionStore) -> None:
        anchor = store._parse_anchor_shorthand("src/handler.py:42")
        assert str(anchor.file_path).endswith("handler.py")
        assert anchor.line_range is not None
        assert anchor.line_range.start == 42
        assert anchor.line_range.end == 42

    def test_file_only(self, store: DecisionStore) -> None:
        anchor = store._parse_anchor_shorthand("src/handler.py")
        assert str(anchor.file_path).endswith("handler.py")
        assert anchor.line_range is None

    def test_file_with_non_numeric_after_colon(
        self, store: DecisionStore,
    ) -> None:
        """If the part after ':' is not numeric, treat entire string as file path."""
        anchor = store._parse_anchor_shorthand("src/handler.py:main")
        # The implementation falls back to treating the whole string as a path
        assert anchor.line_range is None

    def test_preserves_path_components(self, store: DecisionStore) -> None:
        anchor = store._parse_anchor_shorthand("src/api/v2/handler.py:1-100")
        path_str = str(anchor.file_path)
        assert "api" in path_str
        assert "v2" in path_str
        assert anchor.line_range is not None
        assert anchor.line_range.start == 1
        assert anchor.line_range.end == 100
