"""Integration tests for the full graph pipeline: parse -> build -> query.

Creates realistic Python projects in tmp_path, runs GraphBuilder.full_build
and incremental_build, then verifies node/edge creation, impact analysis,
and cleanup of deleted files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codebase_intel.core.config import GraphConfig, ParserConfig, ProjectConfig

if TYPE_CHECKING:
    from pathlib import Path
from codebase_intel.core.types import NodeKind
from codebase_intel.graph.builder import GraphBuilder
from codebase_intel.graph.storage import GraphStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_project_config(project_root: Path, db_path: Path) -> ProjectConfig:
    """Build a ProjectConfig whose graph DB lives in *db_path*."""
    return ProjectConfig(
        project_root=project_root,
        parser=ParserConfig(
            ignored_patterns=["__pycache__/**", ".git/**"],
        ),
        graph=GraphConfig(db_path=db_path),
    )


# ---------------------------------------------------------------------------
# Realistic mini-project layout
# ---------------------------------------------------------------------------


# Source files are deliberately free of external (stdlib) imports so that
# the FOREIGN KEY constraint on edges is never violated — every edge
# target references a node that exists in the graph.
# Files are named so alphabetical discovery order satisfies import deps:
#   a_models.py  (no imports)
#   b_service.py (imports a_models)
#   c_routes.py  (imports b_service)
#   d_utils.py   (no imports — isolated helper)

A_MODELS_PY = """\
class User:
    \"\"\"Domain model for a user.\"\"\"

    def __init__(self, name: str, email: str) -> None:
        self.name = name
        self.email = email


class Order:
    \"\"\"Domain model for an order.\"\"\"

    def __init__(self, user: User, amount: float) -> None:
        self.user = user
        self.amount = amount
"""

B_SERVICE_PY = """\
from a_models import User, Order


def create_user(name: str, email: str) -> User:
    return User(name, email)


def place_order(user: User, amount: float) -> Order:
    return Order(user, amount)
"""

C_ROUTES_PY = """\
from b_service import create_user, place_order


def handle_create_user(payload: dict) -> dict:
    user = create_user(payload["name"], payload["email"])
    return {"id": 1, "name": user.name}


def handle_place_order(payload: dict) -> dict:
    order = place_order(None, payload["amount"])
    return {"order_amount": order.amount}
"""

D_UTILS_PY = """\
def hash_string(value: str) -> str:
    total = 0
    for ch in value:
        total = (total * 31 + ord(ch)) & 0xFFFFFFFF
    return hex(total)
"""


def _create_project(root: Path) -> dict[str, Path]:
    """Write the mini-project files and return a name -> path mapping."""
    files = {
        "models": _write(root / "a_models.py", A_MODELS_PY),
        "service": _write(root / "b_service.py", B_SERVICE_PY),
        "routes": _write(root / "c_routes.py", C_ROUTES_PY),
        "utils": _write(root / "d_utils.py", D_UTILS_PY),
    }
    return files


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullBuildPipeline:
    """Run full_build on a small project and verify the resulting graph."""

    async def test_full_build_creates_nodes_for_all_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.full_build()

        assert progress.processed >= 4  # at least 4 files
        assert progress.nodes_created >= 4  # at least one MODULE per file

    async def test_full_build_creates_module_nodes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            # Each .py file should have a MODULE node
            for name, fp in files.items():
                nodes = await storage.get_nodes_by_file(fp)
                module_nodes = [n for n in nodes if n.kind == NodeKind.MODULE]
                assert len(module_nodes) == 1, (
                    f"Expected 1 MODULE node for {name}.py, got {len(module_nodes)}"
                )

    async def test_full_build_creates_class_nodes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            models_nodes = await storage.get_nodes_by_file(files["models"])
            class_nodes = [n for n in models_nodes if n.kind == NodeKind.CLASS]
            class_names = {n.name for n in class_nodes}
            assert "User" in class_names
            assert "Order" in class_names

    async def test_full_build_creates_function_nodes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            service_nodes = await storage.get_nodes_by_file(files["service"])
            func_nodes = [n for n in service_nodes if n.kind == NodeKind.FUNCTION]
            func_names = {n.name for n in func_nodes}
            assert "create_user" in func_names
            assert "place_order" in func_names

    async def test_full_build_creates_import_edges(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.full_build()
            assert progress.edges_created > 0

    async def test_full_build_stats_consistent(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.full_build()

            stats = await storage.get_stats()
            assert stats["nodes_count"] == progress.nodes_created
            assert stats["edges_count"] == progress.edges_created
            assert stats["file_fingerprints_count"] == progress.processed


class TestImpactAnalysis:
    """Verify impact_analysis finds dependents after a full build."""

    async def test_impact_of_models_change(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            # Changing models.py should affect service.py and routes.py
            result = await storage.impact_analysis(
                [files["models"]], max_depth=3
            )

        affected = result.get(str(files["models"]), [])
        # service.py imports models directly; routes.py imports service
        # At minimum, service.py should appear
        assert len(affected) > 0, (
            "Expected at least one affected node when models.py changes"
        )

    async def test_impact_of_isolated_file(self, tmp_path: Path) -> None:
        """utils.py has no dependents => impact analysis returns empty."""
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            result = await storage.impact_analysis(
                [files["utils"]], max_depth=3
            )

        affected = result.get(str(files["utils"]), [])
        assert len(affected) == 0


class TestIncrementalBuild:
    """Verify incremental builds only re-parse changed files."""

    async def test_incremental_build_only_processes_changed_file(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        # Full build first
        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

        # Modify only d_utils.py
        modified_utils = (
            "def hash_string(value: str) -> str:\n"
            "    total = 0\n"
            "    for ch in value:\n"
            "        total = (total * 31 + ord(ch)) & 0xFFFFFFFF\n"
            "    return hex(total)\n\n"
            "def hmac_sign(key: str, msg: str) -> str:\n"
            "    return hash_string(key + msg)\n"
        )
        _write(files["utils"], modified_utils)

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.incremental_build(
                changed_files=[files["utils"]]
            )

        # Only utils.py was re-processed
        assert progress.processed == 1
        assert progress.total_files == 1

    async def test_incremental_build_updates_nodes(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

        # Add a new function to d_utils.py
        new_utils = D_UTILS_PY + "\ndef new_helper() -> None:\n    pass\n"
        _write(files["utils"], new_utils)

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.incremental_build(changed_files=[files["utils"]])

            utils_nodes = await storage.get_nodes_by_file(files["utils"])
            func_names = {
                n.name for n in utils_nodes if n.kind == NodeKind.FUNCTION
            }
            assert "new_helper" in func_names
            assert "hash_string" in func_names

    async def test_incremental_build_handles_deleted_file(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

        # Delete utils.py
        files["utils"].unlink()

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.incremental_build(
                changed_files=[files["utils"]]
            )

            # File was deleted, so its nodes should be gone
            nodes = await storage.get_nodes_by_file(files["utils"])
            assert len(nodes) == 0

    async def test_incremental_build_preserves_untouched_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            # Record models.py node count
            models_nodes_before = await storage.get_nodes_by_file(files["models"])

        # Only touch d_utils.py
        _write(files["utils"], D_UTILS_PY + "\nz = 1\n")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.incremental_build(changed_files=[files["utils"]])

            # models.py should be untouched
            models_nodes_after = await storage.get_nodes_by_file(files["models"])
            assert len(models_nodes_after) == len(models_nodes_before)


class TestCleanupDeletedFiles:
    """Verify that cleanup_deleted_files removes orphaned nodes."""

    async def test_cleanup_removes_nodes_for_deleted_files(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "project"
        root.mkdir()
        files = _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            await builder.full_build()

            # Verify utils.py has nodes
            nodes_before = await storage.get_nodes_by_file(files["utils"])
            assert len(nodes_before) > 0

        # Delete the file on disk
        files["utils"].unlink()

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            # incremental_build with no explicit changed_files triggers
            # _detect_changed_files + _cleanup_deleted_files
            await builder.incremental_build()

            # Nodes for the deleted file should be gone
            nodes_after = await storage.get_nodes_by_file(files["utils"])
            assert len(nodes_after) == 0


class TestBuildProgress:
    """Verify progress tracking during builds."""

    async def test_progress_summary(self, tmp_path: Path) -> None:
        root = tmp_path / "project"
        root.mkdir()
        _create_project(root)
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.full_build()

        summary = progress.summary()
        assert summary["total_files"] >= 4
        assert summary["processed"] >= 4
        assert summary["nodes_created"] >= 4
        assert summary["failed"] == 0

    async def test_empty_project_completes_without_error(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "empty_project"
        root.mkdir()
        config = _make_project_config(root, tmp_path / "graph.db")

        async with GraphStorage.open(config.graph, root) as storage:
            builder = GraphBuilder(config, storage)
            progress = await builder.full_build()

        assert progress.total_files == 0
        assert progress.processed == 0
        assert progress.completed_pct == 100.0
