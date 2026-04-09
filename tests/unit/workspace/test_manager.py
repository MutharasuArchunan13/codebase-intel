"""Tests for the workspace manager — multi-project routing and LRU eviction."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from codebase_intel.workspace.manager import ProjectState, WorkspaceManager
from codebase_intel.workspace.registry import GlobalRegistry

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def global_dir(tmp_path: Path) -> Path:
    return tmp_path / ".codebase-intel"


@pytest.fixture()
def registry(global_dir: Path) -> GlobalRegistry:
    return GlobalRegistry(global_dir=global_dir)


@pytest.fixture()
def project_a(tmp_path: Path) -> Path:
    proj = tmp_path / "project-a"
    proj.mkdir()
    (proj / ".git").mkdir()
    return proj


@pytest.fixture()
def project_b(tmp_path: Path) -> Path:
    proj = tmp_path / "project-b"
    proj.mkdir()
    (proj / ".git").mkdir()
    return proj


@pytest.fixture()
def project_c(tmp_path: Path) -> Path:
    proj = tmp_path / "project-c"
    proj.mkdir()
    (proj / ".git").mkdir()
    return proj


class TestProjectState:
    def test_initial_state(self, project_a: Path) -> None:
        state = ProjectState(project_a)
        assert state.is_initialized is False
        assert state.state == {}

    @pytest.mark.asyncio()
    async def test_close_clears_state(self, project_a: Path) -> None:
        state = ProjectState(project_a)
        state._state["some_key"] = "value"
        state._initialized = True

        await state.close()

        assert state.is_initialized is False
        assert state.state == {}

    @pytest.mark.asyncio()
    async def test_close_handles_missing_resources(self, project_a: Path) -> None:
        """Close should not raise even if resources don't exist."""
        state = ProjectState(project_a)
        await state.close()  # Should not raise


class TestWorkspaceManager:
    def test_loaded_projects_initially_empty(self, registry: GlobalRegistry) -> None:
        manager = WorkspaceManager(registry=registry)
        assert manager.loaded_projects == []

    @pytest.mark.asyncio()
    async def test_resolve_by_project_id(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        registry.register(project_a)
        manager = WorkspaceManager(registry=registry)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}
            state = await manager.resolve(project_id="project-a")

        assert "error" not in state

    @pytest.mark.asyncio()
    async def test_resolve_unknown_project_id(self, registry: GlobalRegistry) -> None:
        manager = WorkspaceManager(registry=registry)
        state = await manager.resolve(project_id="nonexistent")
        assert "error" in state
        assert "not found" in state["error"]

    @pytest.mark.asyncio()
    async def test_resolve_by_file_path(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        registry.register(project_a)
        manager = WorkspaceManager(registry=registry)

        file_in_project = str(project_a / "src" / "main.py")

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}
            state = await manager.resolve(file_path=file_in_project)

        assert "error" not in state

    @pytest.mark.asyncio()
    async def test_resolve_no_args_returns_error(self, registry: GlobalRegistry) -> None:
        manager = WorkspaceManager(registry=registry)
        state = await manager.resolve()
        assert "error" in state
        assert "Cannot determine project" in state["error"]

    @pytest.mark.asyncio()
    async def test_resolve_missing_root_returns_error(
        self, registry: GlobalRegistry, tmp_path: Path
    ) -> None:
        """Project root was deleted after registration."""
        proj = tmp_path / "deleted-project"
        proj.mkdir()
        registry.register(proj)
        proj.rmdir()  # Remove after registration

        manager = WorkspaceManager(registry=registry)
        state = await manager.resolve(project_id="deleted-project")
        assert "error" in state
        assert "no longer exists" in state["error"]

    @pytest.mark.asyncio()
    async def test_auto_discover_unregistered_project(
        self, registry: GlobalRegistry, tmp_path: Path
    ) -> None:
        """File path not under any registered project → walk up to discover."""
        # Create unregistered project with .git
        proj = tmp_path / "unregistered"
        proj.mkdir()
        (proj / ".git").mkdir()
        (proj / "src").mkdir()

        manager = WorkspaceManager(registry=registry)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}
            state = await manager.resolve(file_path=str(proj / "src" / "app.py"))

        assert "error" not in state
        # Should have been auto-registered
        assert registry.get("unregistered") is not None

    @pytest.mark.asyncio()
    async def test_lru_eviction(
        self,
        registry: GlobalRegistry,
        project_a: Path,
        project_b: Path,
        project_c: Path,
    ) -> None:
        """When max_loaded is exceeded, LRU project is evicted."""
        registry.register(project_a)
        registry.register(project_b)
        registry.register(project_c)

        manager = WorkspaceManager(registry=registry, max_loaded=2)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}

            # Load A and B
            await manager.resolve(project_id="project-a")
            await manager.resolve(project_id="project-b")
            assert len(manager.loaded_projects) == 2

            # Load C → should evict A (LRU)
            await manager.resolve(project_id="project-c")
            assert len(manager.loaded_projects) == 2
            assert "project-a" not in manager.loaded_projects
            assert "project-b" in manager.loaded_projects
            assert "project-c" in manager.loaded_projects

    @pytest.mark.asyncio()
    async def test_lru_reuse_moves_to_end(
        self,
        registry: GlobalRegistry,
        project_a: Path,
        project_b: Path,
        project_c: Path,
    ) -> None:
        """Accessing a loaded project moves it to the end of LRU."""
        registry.register(project_a)
        registry.register(project_b)
        registry.register(project_c)

        manager = WorkspaceManager(registry=registry, max_loaded=2)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}

            await manager.resolve(project_id="project-a")
            await manager.resolve(project_id="project-b")

            # Re-access A → A becomes most recently used
            await manager.resolve(project_id="project-a")

            # Load C → should evict B (now LRU), not A
            await manager.resolve(project_id="project-c")
            assert "project-a" in manager.loaded_projects
            assert "project-b" not in manager.loaded_projects
            assert "project-c" in manager.loaded_projects

    @pytest.mark.asyncio()
    async def test_close_all(
        self, registry: GlobalRegistry, project_a: Path, project_b: Path
    ) -> None:
        registry.register(project_a)
        registry.register(project_b)

        manager = WorkspaceManager(registry=registry)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}
            await manager.resolve(project_id="project-a")
            await manager.resolve(project_id="project-b")

        await manager.close_all()
        assert manager.loaded_projects == []

    def test_discover_project_root_git(self, tmp_path: Path) -> None:
        proj = tmp_path / "my-project"
        proj.mkdir()
        (proj / ".git").mkdir()
        deep = proj / "src" / "lib" / "utils"
        deep.mkdir(parents=True)

        manager = WorkspaceManager()
        root = manager._discover_project_root(deep)
        assert root == proj

    def test_discover_project_root_pyproject(self, tmp_path: Path) -> None:
        proj = tmp_path / "py-project"
        proj.mkdir()
        (proj / "pyproject.toml").write_text("[build-system]")
        deep = proj / "src"
        deep.mkdir()

        manager = WorkspaceManager()
        root = manager._discover_project_root(deep)
        assert root == proj

    def test_discover_project_root_package_json(self, tmp_path: Path) -> None:
        proj = tmp_path / "node-project"
        proj.mkdir()
        (proj / "package.json").write_text("{}")
        deep = proj / "src"
        deep.mkdir()

        manager = WorkspaceManager()
        root = manager._discover_project_root(deep)
        assert root == proj

    def test_discover_project_root_cargo(self, tmp_path: Path) -> None:
        proj = tmp_path / "rust-project"
        proj.mkdir()
        (proj / "Cargo.toml").write_text("[package]")

        manager = WorkspaceManager()
        root = manager._discover_project_root(proj)
        assert root == proj

    def test_discover_no_root_found(self, tmp_path: Path) -> None:
        """Directory with no project markers returns None."""
        bare = tmp_path / "bare-dir"
        bare.mkdir()

        manager = WorkspaceManager()
        # Could find tmp_path markers from other fixtures, so test a truly bare path
        root = manager._discover_project_root(bare / "nonexistent")
        assert root is None

    @pytest.mark.asyncio()
    async def test_resolve_injects_project_metadata(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        registry.register(project_a, name="My Project A")
        manager = WorkspaceManager(registry=registry)

        with patch.object(ProjectState, "ensure_initialized", new_callable=AsyncMock) as mock_init:
            mock_init.return_value = {"initialized": True, "config": MagicMock()}
            state = await manager.resolve(project_id="project-a")

        assert state["_project_id"] == "project-a"
        assert state["_project_name"] == "My Project A"
