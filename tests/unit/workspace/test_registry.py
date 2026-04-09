"""Tests for the global project registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from codebase_intel.workspace.registry import GlobalRegistry, ProjectEntry


@pytest.fixture()
def global_dir(tmp_path: Path) -> Path:
    """Temp directory for global registry."""
    return tmp_path / ".codebase-intel"


@pytest.fixture()
def registry(global_dir: Path) -> GlobalRegistry:
    return GlobalRegistry(global_dir=global_dir)


@pytest.fixture()
def project_a(tmp_path: Path) -> Path:
    """Create a fake project directory."""
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
def initialized_project(tmp_path: Path) -> Path:
    """Project with .codebase-intel/graph.db."""
    proj = tmp_path / "initialized-proj"
    proj.mkdir()
    intel_dir = proj / ".codebase-intel"
    intel_dir.mkdir()
    (intel_dir / "graph.db").write_text("fake db")
    return proj


class TestProjectEntry:
    def test_is_valid_existing_dir(self, project_a: Path) -> None:
        entry = ProjectEntry(project_id="a", root_path=str(project_a))
        assert entry.is_valid is True

    def test_is_valid_missing_dir(self) -> None:
        entry = ProjectEntry(project_id="x", root_path="/nonexistent/path")
        assert entry.is_valid is False

    def test_root_property(self, project_a: Path) -> None:
        entry = ProjectEntry(project_id="a", root_path=str(project_a))
        assert entry.root == project_a


class TestGlobalRegistry:
    def test_register_project(self, registry: GlobalRegistry, project_a: Path) -> None:
        entry = registry.register(project_a)
        assert entry.project_id == "project-a"
        assert entry.root_path == str(project_a)
        assert entry.initialized is False

    def test_register_with_custom_id(self, registry: GlobalRegistry, project_a: Path) -> None:
        entry = registry.register(project_a, project_id="my-svc", name="My Service")
        assert entry.project_id == "my-svc"
        assert entry.name == "My Service"

    def test_register_initialized_project(
        self, registry: GlobalRegistry, initialized_project: Path
    ) -> None:
        entry = registry.register(initialized_project)
        assert entry.initialized is True

    def test_register_nonexistent_raises(self, registry: GlobalRegistry) -> None:
        with pytest.raises(ValueError, match="does not exist"):
            registry.register(Path("/nonexistent/project"))

    def test_register_idempotent(self, registry: GlobalRegistry, project_a: Path) -> None:
        entry1 = registry.register(project_a)
        entry2 = registry.register(project_a, name="Updated Name")
        assert entry2.name == "Updated Name"
        # Keeps original registration timestamp
        assert entry1.registered_at == entry2.registered_at

    def test_unregister(self, registry: GlobalRegistry, project_a: Path) -> None:
        registry.register(project_a)
        assert registry.unregister("project-a") is True
        assert registry.get("project-a") is None

    def test_unregister_nonexistent(self, registry: GlobalRegistry) -> None:
        assert registry.unregister("nope") is False

    def test_get(self, registry: GlobalRegistry, project_a: Path) -> None:
        registry.register(project_a)
        entry = registry.get("project-a")
        assert entry is not None
        assert entry.project_id == "project-a"

    def test_get_nonexistent(self, registry: GlobalRegistry) -> None:
        assert registry.get("nope") is None

    def test_get_all(
        self, registry: GlobalRegistry, project_a: Path, project_b: Path
    ) -> None:
        registry.register(project_a)
        registry.register(project_b)
        all_projects = registry.get_all()
        assert len(all_projects) == 2
        ids = {p.project_id for p in all_projects}
        assert ids == {"project-a", "project-b"}

    def test_get_valid_excludes_missing(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        registry.register(project_a)
        # Manually add a project with nonexistent path
        from codebase_intel.workspace.registry import ProjectEntry

        registry._projects["ghost"] = ProjectEntry(
            project_id="ghost", root_path="/nonexistent"
        )
        registry._save()

        valid = registry.get_valid()
        assert len(valid) == 1
        assert valid[0].project_id == "project-a"

    def test_find_by_path_direct_match(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        registry.register(project_a)
        entry = registry.find_by_path(project_a / "src" / "main.py")
        assert entry is not None
        assert entry.project_id == "project-a"

    def test_find_by_path_no_match(self, registry: GlobalRegistry) -> None:
        assert registry.find_by_path(Path("/random/file.py")) is None

    def test_find_by_path_picks_deepest(
        self, registry: GlobalRegistry, tmp_path: Path
    ) -> None:
        """Nested projects: pick the most specific match."""
        parent = tmp_path / "monorepo"
        parent.mkdir()
        child = parent / "services" / "auth"
        child.mkdir(parents=True)
        (child / ".git").mkdir()

        registry.register(parent, project_id="monorepo")
        registry.register(child, project_id="auth-svc")

        # File inside auth should match auth-svc, not monorepo
        entry = registry.find_by_path(child / "src" / "handler.py")
        assert entry is not None
        assert entry.project_id == "auth-svc"

    def test_persistence_across_instances(
        self, global_dir: Path, project_a: Path
    ) -> None:
        """Registry survives restart (new instance reads from disk)."""
        reg1 = GlobalRegistry(global_dir=global_dir)
        reg1.register(project_a)

        reg2 = GlobalRegistry(global_dir=global_dir)
        assert reg2.get("project-a") is not None
        assert reg2.get("project-a").root_path == str(project_a)

    def test_get_all_paths(
        self, registry: GlobalRegistry, project_a: Path, project_b: Path
    ) -> None:
        registry.register(project_a)
        registry.register(project_b)
        paths = registry.get_all_paths()
        assert len(paths) == 2
        assert project_a in paths
        assert project_b in paths

    def test_touch_updates_last_accessed(
        self, registry: GlobalRegistry, project_a: Path
    ) -> None:
        entry = registry.register(project_a)
        before = entry.last_accessed

        registry.touch("project-a")

        refreshed = registry.get("project-a")
        assert refreshed is not None
        assert refreshed.last_accessed is not None
        if before:
            assert refreshed.last_accessed >= before

    def test_env_var_override(
        self, tmp_path: Path, project_a: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        custom_dir = tmp_path / "custom-intel-home"
        monkeypatch.setenv("CODEBASE_INTEL_HOME", str(custom_dir))

        reg = GlobalRegistry()
        reg.register(project_a)

        assert (custom_dir / "registry.yaml").exists()

    def test_corrupt_registry_file(self, global_dir: Path) -> None:
        """Corrupted YAML doesn't crash — starts fresh."""
        global_dir.mkdir(parents=True, exist_ok=True)
        (global_dir / "registry.yaml").write_text("{{invalid yaml: [")

        reg = GlobalRegistry(global_dir=global_dir)
        assert reg.get_all() == []
