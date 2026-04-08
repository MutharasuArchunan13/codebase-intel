"""Tests for GraphStorage — SQLite-backed graph storage with WAL mode.

Covers: schema creation, node CRUD, edge CRUD, traversal, impact analysis,
fingerprint operations, cascading deletes, and graph statistics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_intel.core.config import GraphConfig
from codebase_intel.core.types import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    Language,
    LineRange,
    NodeKind,
)
from codebase_intel.graph.storage import SCHEMA_VERSION, GraphStorage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph_config(tmp_path: Path) -> GraphConfig:
    """Produce a GraphConfig pointing at an ephemeral SQLite file."""
    return GraphConfig(db_path=tmp_path / "test.db")


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """Create a minimal project root directory."""
    root = tmp_path / "project"
    root.mkdir()
    return root


def _make_node(
    project_root: Path,
    *,
    name: str = "my_func",
    kind: NodeKind = NodeKind.FUNCTION,
    file_name: str = "module.py",
    language: Language = Language.PYTHON,
    line_start: int = 1,
    line_end: int = 10,
    is_test: bool = False,
    is_generated: bool = False,
    is_entry_point: bool = False,
    docstring: str | None = None,
) -> GraphNode:
    """Build a GraphNode anchored inside *project_root*."""
    file_path = project_root / file_name
    node_id = GraphNode.make_id(file_path, kind, name)
    return GraphNode(
        node_id=node_id,
        kind=kind,
        name=name,
        qualified_name=f"project.{name}",
        file_path=file_path,
        line_range=LineRange(start=line_start, end=line_end),
        language=language,
        content_hash="abc123",
        docstring=docstring,
        is_test=is_test,
        is_generated=is_generated,
        is_entry_point=is_entry_point,
    )


def _make_edge(
    source_id: str,
    target_id: str,
    kind: EdgeKind = EdgeKind.IMPORTS,
    confidence: float = 1.0,
    is_type_only: bool = False,
) -> GraphEdge:
    return GraphEdge(
        source_id=source_id,
        target_id=target_id,
        kind=kind,
        confidence=confidence,
        is_type_only=is_type_only,
    )


# ---------------------------------------------------------------------------
# Schema / open
# ---------------------------------------------------------------------------


class TestGraphStorageOpen:
    """Verify that opening the storage creates the DB, schema, and pragmas."""

    async def test_creates_database_and_schema(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            cursor = await storage._db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = {row[0] for row in await cursor.fetchall()}

        expected_tables = {"schema_version", "build_status", "nodes", "edges", "file_fingerprints"}
        assert expected_tables.issubset(tables)

    async def test_schema_version_recorded(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            cursor = await storage._db.execute("SELECT MAX(version) FROM schema_version")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == SCHEMA_VERSION

    async def test_wal_mode_enabled(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            cursor = await storage._db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == "wal"

    async def test_foreign_keys_enabled(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            cursor = await storage._db.execute("PRAGMA foreign_keys")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1

    async def test_reopening_existing_db_is_idempotent(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """Opening an already-initialised DB should not duplicate rows."""
        async with GraphStorage.open(graph_config, project_root):
            pass
        async with GraphStorage.open(graph_config, project_root) as storage:
            cursor = await storage._db.execute("SELECT COUNT(*) FROM schema_version")
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == 1  # only one version row

    async def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "graph.db"
        config = GraphConfig(db_path=nested)
        root = tmp_path / "proj"
        root.mkdir()
        async with GraphStorage.open(config, root):
            pass
        assert nested.exists()


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------


class TestNodeOperations:
    async def test_upsert_and_get_node_roundtrip(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        node = _make_node(project_root, name="greet", docstring="Say hello")
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node)
            await storage._db.commit()

            fetched = await storage.get_node(node.node_id)

        assert fetched is not None
        assert fetched.node_id == node.node_id
        assert fetched.name == "greet"
        assert fetched.kind == NodeKind.FUNCTION
        assert fetched.language == Language.PYTHON
        assert fetched.docstring == "Say hello"
        assert fetched.line_range is not None
        assert fetched.line_range.start == 1
        assert fetched.line_range.end == 10

    async def test_get_node_returns_none_for_missing(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            assert await storage.get_node("nonexistent") is None

    async def test_upsert_node_updates_existing(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        node_v1 = _make_node(project_root, name="calc", docstring="v1")
        node_v2 = GraphNode(
            node_id=node_v1.node_id,
            kind=node_v1.kind,
            name=node_v1.name,
            qualified_name=node_v1.qualified_name,
            file_path=node_v1.file_path,
            line_range=LineRange(start=5, end=20),
            language=Language.PYTHON,
            content_hash="updated_hash",
            docstring="v2 updated",
        )
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node_v1)
            await storage._db.commit()
            await storage.upsert_node(node_v2)
            await storage._db.commit()

            fetched = await storage.get_node(node_v1.node_id)

        assert fetched is not None
        assert fetched.docstring == "v2 updated"
        assert fetched.line_range is not None
        assert fetched.line_range.start == 5

    async def test_node_without_line_range(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        file_path = project_root / "config.yaml"
        node = GraphNode(
            node_id=GraphNode.make_id(file_path, NodeKind.CONFIG, "config"),
            kind=NodeKind.CONFIG,
            name="config",
            qualified_name="project.config",
            file_path=file_path,
            language=Language.UNKNOWN,
        )
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node)
            await storage._db.commit()
            fetched = await storage.get_node(node.node_id)

        assert fetched is not None
        assert fetched.line_range is None

    async def test_node_boolean_flags_roundtrip(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        node = _make_node(
            project_root,
            name="test_it",
            is_test=True,
            is_generated=True,
            is_entry_point=True,
        )
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node)
            await storage._db.commit()
            fetched = await storage.get_node(node.node_id)

        assert fetched is not None
        assert fetched.is_test is True
        assert fetched.is_generated is True
        assert fetched.is_entry_point is True


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------


class TestBatchOperations:
    async def test_upsert_nodes_batch(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        nodes = [
            _make_node(project_root, name=f"func_{i}", file_name=f"mod_{i}.py")
            for i in range(10)
        ]
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch(nodes)

            for node in nodes:
                fetched = await storage.get_node(node.node_id)
                assert fetched is not None
                assert fetched.name == node.name

    async def test_upsert_nodes_batch_empty_list(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """Batch-upserting an empty list should not raise."""
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch([])

    async def test_upsert_edges_batch(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        nodes = [
            _make_node(project_root, name=f"svc_{i}", file_name=f"svc_{i}.py")
            for i in range(3)
        ]
        edges = [
            _make_edge(nodes[0].node_id, nodes[1].node_id, EdgeKind.IMPORTS),
            _make_edge(nodes[1].node_id, nodes[2].node_id, EdgeKind.CALLS),
        ]
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch(nodes)
            await storage.upsert_edges_batch(edges)

            stats = await storage.get_stats()
            assert stats["edges_count"] == 2


# ---------------------------------------------------------------------------
# Edge operations + traversal
# ---------------------------------------------------------------------------


class TestEdgeTraversal:
    async def _setup_chain(
        self, storage: GraphStorage, project_root: Path
    ) -> list[GraphNode]:
        """Create A -> B -> C chain."""
        a = _make_node(project_root, name="a", file_name="a.py")
        b = _make_node(project_root, name="b", file_name="b.py")
        c = _make_node(project_root, name="c", file_name="c.py")
        for n in (a, b, c):
            await storage.upsert_node(n)
        await storage._db.commit()
        await storage.upsert_edge(_make_edge(a.node_id, b.node_id))
        await storage._db.commit()
        await storage.upsert_edge(_make_edge(b.node_id, c.node_id))
        await storage._db.commit()
        return [a, b, c]

    async def test_get_dependents_single_depth(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            a, b, c = await self._setup_chain(storage, project_root)
            # B's dependents (who imports B?) => A
            deps = await storage.get_dependents(b.node_id, max_depth=1)
            dep_ids = {d.node_id for d in deps}
            assert a.node_id in dep_ids
            assert c.node_id not in dep_ids

    async def test_get_dependents_multi_depth(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            a, b, c = await self._setup_chain(storage, project_root)
            # C's dependents at depth 2 => B (direct) and A (transitive)
            deps = await storage.get_dependents(c.node_id, max_depth=2)
            dep_ids = {d.node_id for d in deps}
            assert b.node_id in dep_ids
            assert a.node_id in dep_ids

    async def test_get_dependencies_forward_traversal(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """get_dependencies at max_depth=1 traverses one hop from the root,
        which discovers B (direct) and also C (found while processing B,
        since B is at depth 1 which equals max_depth). The code adds
        discovered nodes to the result even if it does not enqueue them
        for further traversal.
        """
        async with GraphStorage.open(graph_config, project_root) as storage:
            a, b, c = await self._setup_chain(storage, project_root)
            deps = await storage.get_dependencies(a.node_id, max_depth=1)
            dep_ids = {d.node_id for d in deps}
            assert b.node_id in dep_ids
            # C is also found because B (at depth 1) is still processed
            assert c.node_id in dep_ids

    async def test_get_dependencies_with_max_depth(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            a, b, c = await self._setup_chain(storage, project_root)
            # A depends on B -> C transitively
            deps = await storage.get_dependencies(a.node_id, max_depth=2)
            dep_ids = {d.node_id for d in deps}
            assert b.node_id in dep_ids
            assert c.node_id in dep_ids

    async def test_circular_dependency_does_not_loop(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """A -> B -> C -> A must not infinite-loop."""
        async with GraphStorage.open(graph_config, project_root) as storage:
            a, b, c = await self._setup_chain(storage, project_root)
            # Close the cycle: C -> A
            await storage.upsert_edge(_make_edge(c.node_id, a.node_id))
            await storage._db.commit()

            deps = await storage.get_dependents(a.node_id, max_depth=5)
            dep_ids = {d.node_id for d in deps}
            # Should find B and C without hanging
            assert b.node_id in dep_ids or c.node_id in dep_ids

    async def test_type_only_edges_excluded_by_default(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            a = _make_node(project_root, name="alpha", file_name="alpha.py")
            b = _make_node(project_root, name="beta", file_name="beta.py")
            await storage.upsert_node(a)
            await storage.upsert_node(b)
            await storage._db.commit()
            await storage.upsert_edge(
                _make_edge(a.node_id, b.node_id, is_type_only=True)
            )
            await storage._db.commit()

            # Default: type-only excluded
            deps = await storage.get_dependents(b.node_id)
            assert len(deps) == 0

            # Explicit: include type-only
            deps = await storage.get_dependents(b.node_id, include_type_only=True)
            assert len(deps) == 1

    async def test_edge_kind_filter(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            a = _make_node(project_root, name="x", file_name="x.py")
            b = _make_node(project_root, name="y", file_name="y.py")
            await storage.upsert_node(a)
            await storage.upsert_node(b)
            await storage._db.commit()
            await storage.upsert_edge(_make_edge(a.node_id, b.node_id, EdgeKind.CALLS))
            await storage._db.commit()

            deps = await storage.get_dependents(
                b.node_id, edge_kinds=[EdgeKind.IMPORTS]
            )
            assert len(deps) == 0

            deps = await storage.get_dependents(
                b.node_id, edge_kinds=[EdgeKind.CALLS]
            )
            assert len(deps) == 1


# ---------------------------------------------------------------------------
# Remove file nodes (cascade)
# ---------------------------------------------------------------------------


class TestRemoveFileNodes:
    async def test_remove_deletes_nodes_and_edges(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        file_path = project_root / "doomed.py"
        node_a = _make_node(project_root, name="doom_fn", file_name="doomed.py")
        node_b = _make_node(project_root, name="survivor", file_name="safe.py")

        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node_a)
            await storage.upsert_node(node_b)
            await storage._db.commit()
            await storage.upsert_edge(
                _make_edge(node_b.node_id, node_a.node_id, EdgeKind.IMPORTS)
            )
            await storage._db.commit()

            removed = await storage.remove_file_nodes(file_path)
            assert removed == 1

            # Node gone
            assert await storage.get_node(node_a.node_id) is None
            # Surviving node still present
            assert await storage.get_node(node_b.node_id) is not None
            # Edge also gone (cascade)
            stats = await storage.get_stats()
            assert stats["edges_count"] == 0

    async def test_remove_nonexistent_file_returns_zero(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            removed = await storage.remove_file_nodes(project_root / "ghost.py")
            assert removed == 0


# ---------------------------------------------------------------------------
# Impact analysis
# ---------------------------------------------------------------------------


class TestImpactAnalysis:
    async def test_finds_dependents_of_changed_file(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """If models.py changes, service.py (which imports it) should appear."""
        models_path = project_root / "models.py"

        models_node = _make_node(
            project_root, name="models", kind=NodeKind.MODULE, file_name="models.py"
        )
        service_node = _make_node(
            project_root, name="service", kind=NodeKind.MODULE, file_name="service.py"
        )

        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch([models_node, service_node])
            await storage.upsert_edge(
                _make_edge(service_node.node_id, models_node.node_id, EdgeKind.IMPORTS)
            )
            await storage._db.commit()

            result = await storage.impact_analysis([models_path])

        affected = result.get(str(models_path), [])
        affected_ids = {n.node_id for n in affected}
        assert service_node.node_id in affected_ids

    async def test_impact_analysis_unknown_file_returns_empty(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            result = await storage.impact_analysis([project_root / "new_file.py"])
        assert result[str(project_root / "new_file.py")] == []

    async def test_impact_analysis_transitive(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        """A -> B -> C: changing C should surface A at depth >= 2."""
        nodes = [
            _make_node(project_root, name=n, kind=NodeKind.MODULE, file_name=f"{n}.py")
            for n in ("a_mod", "b_mod", "c_mod")
        ]
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch(nodes)
            await storage.upsert_edge(
                _make_edge(nodes[0].node_id, nodes[1].node_id, EdgeKind.IMPORTS)
            )
            await storage.upsert_edge(
                _make_edge(nodes[1].node_id, nodes[2].node_id, EdgeKind.IMPORTS)
            )
            await storage._db.commit()

            result = await storage.impact_analysis(
                [project_root / "c_mod.py"], max_depth=3
            )

        affected = result[str(project_root / "c_mod.py")]
        affected_ids = {n.node_id for n in affected}
        assert nodes[1].node_id in affected_ids  # b_mod
        assert nodes[0].node_id in affected_ids  # a_mod


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_get_stats_counts(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        nodes = [
            _make_node(project_root, name=f"fn_{i}", file_name=f"f_{i}.py")
            for i in range(5)
        ]
        edge = _make_edge(nodes[0].node_id, nodes[1].node_id)

        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch(nodes)
            await storage.upsert_edge(edge)
            await storage._db.commit()

            stats = await storage.get_stats()

        assert stats["nodes_count"] == 5
        assert stats["edges_count"] == 1
        assert stats["file_fingerprints_count"] == 0

    async def test_get_stats_empty_db(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            stats = await storage.get_stats()
        assert stats["nodes_count"] == 0
        assert stats["edges_count"] == 0


# ---------------------------------------------------------------------------
# Fingerprint operations
# ---------------------------------------------------------------------------


class TestFingerprints:
    async def test_update_and_get_fingerprint(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        fp = project_root / "utils.py"
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.update_fingerprint(
                file_path=fp,
                content_hash="deadbeef",
                size_bytes=1024,
                last_modified="2026-01-01T00:00:00",
                language=Language.PYTHON,
                node_count=5,
            )
            result = await storage.get_fingerprint(fp)

        assert result == "deadbeef"

    async def test_get_fingerprint_missing_returns_none(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            result = await storage.get_fingerprint(project_root / "nope.py")
        assert result is None

    async def test_update_fingerprint_overwrites(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        fp = project_root / "app.py"
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.update_fingerprint(
                file_path=fp,
                content_hash="hash_v1",
                size_bytes=100,
                last_modified="2026-01-01T00:00:00",
                language=Language.PYTHON,
                node_count=2,
            )
            await storage.update_fingerprint(
                file_path=fp,
                content_hash="hash_v2",
                size_bytes=200,
                last_modified="2026-02-01T00:00:00",
                language=Language.PYTHON,
                node_count=4,
            )
            result = await storage.get_fingerprint(fp)

        assert result == "hash_v2"

    async def test_fingerprint_counted_in_stats(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.update_fingerprint(
                file_path=project_root / "one.py",
                content_hash="h1",
                size_bytes=50,
                last_modified="2026-01-01T00:00:00",
                language=Language.PYTHON,
                node_count=1,
            )
            stats = await storage.get_stats()
        assert stats["file_fingerprints_count"] == 1


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------


class TestPathNormalisation:
    async def test_nodes_stored_with_posix_relative_paths(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        sub = project_root / "pkg" / "deep.py"
        node = GraphNode(
            node_id=GraphNode.make_id(sub, NodeKind.MODULE, "deep"),
            kind=NodeKind.MODULE,
            name="deep",
            qualified_name="pkg.deep",
            file_path=sub,
            language=Language.PYTHON,
        )
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_node(node)
            await storage._db.commit()

            cursor = await storage._db.execute(
                "SELECT file_path FROM nodes WHERE node_id = ?", (node.node_id,)
            )
            row = await cursor.fetchone()

        assert row is not None
        stored_path = row[0]
        assert "/" in stored_path  # POSIX separator
        assert not Path(stored_path).is_absolute()

    async def test_get_nodes_by_file(
        self, graph_config: GraphConfig, project_root: Path
    ) -> None:
        fp = project_root / "views.py"
        n1 = _make_node(project_root, name="view_a", file_name="views.py")
        n2 = _make_node(
            project_root, name="view_b", kind=NodeKind.CLASS, file_name="views.py"
        )
        async with GraphStorage.open(graph_config, project_root) as storage:
            await storage.upsert_nodes_batch([n1, n2])
            nodes = await storage.get_nodes_by_file(fp)

        assert len(nodes) == 2
        names = {n.name for n in nodes}
        assert names == {"view_a", "view_b"}
