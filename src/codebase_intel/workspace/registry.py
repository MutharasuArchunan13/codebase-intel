"""Global registry — persistent catalog of all known projects.

Stored at ~/.codebase-intel/registry.yaml so it survives across sessions.
Each entry maps a project_id to its root path and metadata.

Edge cases:
- Registry file doesn't exist: create on first register
- Project path moved/deleted: detected on load, marked stale
- Duplicate registration: update timestamp, don't duplicate
- Permission errors on ~/.codebase-intel: fall back to /tmp
- Concurrent writes: YAML is small, atomic write via temp file + rename
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_DIR = Path.home() / ".codebase-intel"


class ProjectEntry(BaseModel):
    """A registered project in the global catalog."""

    project_id: str
    root_path: str = Field(description="Absolute path to project root")
    name: str = ""
    registered_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_accessed: datetime | None = None
    initialized: bool = Field(
        default=False,
        description="True if .codebase-intel/ exists with graph.db",
    )

    @property
    def root(self) -> Path:
        return Path(self.root_path)

    @property
    def is_valid(self) -> bool:
        """Check if the project root still exists on disk."""
        return self.root.is_dir()


class GlobalRegistry:
    """Persistent global catalog of all known projects.

    Location: ~/.codebase-intel/registry.yaml (configurable via
    CODEBASE_INTEL_HOME env var).
    """

    def __init__(self, global_dir: Path | None = None) -> None:
        env_home = os.environ.get("CODEBASE_INTEL_HOME")
        if env_home:
            self._global_dir = Path(env_home)
        elif global_dir:
            self._global_dir = global_dir
        else:
            self._global_dir = DEFAULT_GLOBAL_DIR

        self._global_dir.mkdir(parents=True, exist_ok=True)
        self._registry_path = self._global_dir / "registry.yaml"
        self._projects: dict[str, ProjectEntry] = {}
        self._load()

    @property
    def global_dir(self) -> Path:
        return self._global_dir

    def _load(self) -> None:
        """Load registry from disk."""
        if not self._registry_path.exists():
            self._projects = {}
            return

        try:
            raw = yaml.safe_load(self._registry_path.read_text(encoding="utf-8"))
            if not raw or "projects" not in raw:
                self._projects = {}
                return

            for pid, data in raw["projects"].items():
                self._projects[pid] = ProjectEntry(
                    project_id=pid,
                    root_path=data["root_path"],
                    name=data.get("name", ""),
                    registered_at=datetime.fromisoformat(data["registered_at"])
                    if data.get("registered_at")
                    else datetime.now(UTC),
                    last_accessed=datetime.fromisoformat(data["last_accessed"])
                    if data.get("last_accessed")
                    else None,
                    initialized=data.get("initialized", False),
                )
        except Exception:
            logger.warning("Failed to load registry at %s, starting fresh", self._registry_path)
            self._projects = {}

    def _save(self) -> None:
        """Persist registry to disk via atomic write."""
        data: dict[str, Any] = {"projects": {}}
        for pid, entry in self._projects.items():
            data["projects"][pid] = {
                "root_path": entry.root_path,
                "name": entry.name,
                "registered_at": entry.registered_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat()
                if entry.last_accessed
                else None,
                "initialized": entry.initialized,
            }

        self._global_dir.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._global_dir), suffix=".yaml.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            os.replace(tmp_path, str(self._registry_path))
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def register(
        self,
        root_path: Path,
        project_id: str | None = None,
        name: str | None = None,
    ) -> ProjectEntry:
        """Register a project in the global catalog.

        If project_id is not given, derives it from the directory name.
        If the project is already registered, updates the entry.
        """
        root_path = root_path.resolve()
        if not root_path.is_dir():
            msg = f"Project root does not exist: {root_path}"
            raise ValueError(msg)

        pid = project_id or root_path.name
        display_name = name or pid.replace("-", " ").replace("_", " ").title()

        # Check if .codebase-intel is initialized
        intel_dir = root_path / ".codebase-intel"
        is_initialized = (intel_dir / "graph.db").exists()

        entry = ProjectEntry(
            project_id=pid,
            root_path=str(root_path),
            name=display_name,
            registered_at=self._projects[pid].registered_at
            if pid in self._projects
            else datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            initialized=is_initialized,
        )
        self._projects[pid] = entry
        self._save()

        logger.info("Registered project '%s' at %s", pid, root_path)
        return entry

    def unregister(self, project_id: str) -> bool:
        """Remove a project from the global catalog."""
        if project_id not in self._projects:
            return False

        del self._projects[project_id]
        self._save()
        logger.info("Unregistered project '%s'", project_id)
        return True

    def get(self, project_id: str) -> ProjectEntry | None:
        """Get a registered project by ID."""
        return self._projects.get(project_id)

    def get_all(self) -> list[ProjectEntry]:
        """Get all registered projects."""
        return list(self._projects.values())

    def get_valid(self) -> list[ProjectEntry]:
        """Get all registered projects whose root paths still exist."""
        return [p for p in self._projects.values() if p.is_valid]

    def find_by_path(self, file_path: Path) -> ProjectEntry | None:
        """Find which registered project a file path belongs to.

        Walks through all registered projects and checks if
        file_path is under any project's root. Returns the most
        specific match (longest root path).

        Edge cases:
        - File under multiple registered roots (nested projects): pick deepest
        - File not under any registered root: return None
        - Symlinks: resolve before matching
        """
        file_path = file_path.resolve()
        best_match: ProjectEntry | None = None
        best_depth = 0

        for entry in self._projects.values():
            root = entry.root
            try:
                file_path.relative_to(root)
                depth = len(root.parts)
                if depth > best_depth:
                    best_match = entry
                    best_depth = depth
            except ValueError:
                continue

        if best_match:
            # Update last_accessed
            best_match.last_accessed = datetime.now(UTC)
            self._save()

        return best_match

    def touch(self, project_id: str) -> None:
        """Update last_accessed timestamp for a project."""
        entry = self._projects.get(project_id)
        if entry:
            entry.last_accessed = datetime.now(UTC)
            self._save()

    def get_all_paths(self) -> list[Path]:
        """Get root paths for all valid registered projects."""
        return [entry.root for entry in self._projects.values() if entry.is_valid]
