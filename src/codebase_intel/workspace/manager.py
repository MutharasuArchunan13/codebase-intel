"""Workspace manager — multi-project state with lazy loading and LRU eviction.

The core problem: codebase-intel's MCP server was designed as 1 server = 1 project.
This module enables 1 server = N projects by:

1. Maintaining lazy-loaded ProjectState instances (one per project)
2. Routing requests to the correct project based on file paths
3. Evicting least-recently-used projects to bound memory
4. Auto-discovering project roots when file paths don't match any registered project

Edge cases:
- File path matches no registered project: walk up to find .git, auto-register
- Project not initialized (.codebase-intel missing): return degraded state
- Too many projects loaded: LRU eviction closes SQLite connections
- Project root deleted while server running: detect and clean up
- Concurrent access to same project state: safe because SQLite uses WAL
"""

from __future__ import annotations

import contextlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from codebase_intel.workspace.registry import GlobalRegistry, ProjectEntry

logger = logging.getLogger(__name__)


class ProjectState:
    """Lazy-loaded state for a single project.

    Components are initialized on first access, not at construction.
    This keeps startup fast when many projects are registered.
    """

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        self._initialized = False
        self._state: dict[str, Any] = {}

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    async def ensure_initialized(self) -> dict[str, Any]:
        """Initialize all components lazily. Mirrors the logic from server.py."""
        if self._initialized:
            return self._state

        from codebase_intel.core.config import ProjectConfig

        try:
            config = ProjectConfig(project_root=self.project_root)
        except Exception as exc:
            self._state["error"] = f"Configuration error: {exc}"
            self._initialized = True
            return self._state

        self._state["config"] = config

        # Graph
        try:
            from codebase_intel.graph.storage import GraphStorage

            if config.graph.db_path.exists():
                import aiosqlite

                db = await aiosqlite.connect(str(config.graph.db_path))
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=5000")
                await db.execute("PRAGMA foreign_keys=ON")
                storage = GraphStorage(db, config.project_root)
                await storage._ensure_schema()

                from codebase_intel.graph.query import GraphQueryEngine

                self._state["graph_storage"] = storage
                self._state["graph_engine"] = GraphQueryEngine(storage)
                self._state["graph_db"] = db
                self._state["graph_available"] = True
            else:
                self._state["graph_available"] = False
        except Exception as exc:
            logger.warning("Graph init failed for %s: %s", self.project_root, exc)
            self._state["graph_available"] = False

        # Decisions
        try:
            from codebase_intel.decisions.store import DecisionStore

            store = DecisionStore(config.decisions, config.project_root)
            self._state["decisions"] = store
            self._state["decisions_available"] = True
        except Exception:
            self._state["decisions_available"] = False

        # Contracts
        try:
            from codebase_intel.contracts.registry import ContractRegistry

            registry = ContractRegistry(config.contracts, config.project_root)
            registry.load()
            self._state["contracts"] = registry
            self._state["contracts_available"] = True
        except Exception:
            self._state["contracts_available"] = False

        # Analytics
        try:
            from codebase_intel.analytics.tracker import AnalyticsTracker

            analytics_db = config.project_root / ".codebase-intel" / "analytics.db"
            self._state["analytics"] = AnalyticsTracker(analytics_db)
            self._state["analytics_available"] = True
        except Exception:
            self._state["analytics_available"] = False

        # Feedback
        try:
            from codebase_intel.analytics.feedback import FeedbackTracker

            feedback_db = config.project_root / ".codebase-intel" / "feedback.db"
            self._state["feedback"] = FeedbackTracker(feedback_db)
            self._state["feedback_available"] = True
        except Exception:
            self._state["feedback_available"] = False

        # Intents
        try:
            from codebase_intel.intent.store import IntentStore

            intents_dir = config.project_root / ".codebase-intel" / "intents"
            self._state["intent_store"] = IntentStore(intents_dir)
            self._state["intent_available"] = True
        except Exception:
            self._state["intent_available"] = False

        self._state["initialized"] = True
        self._initialized = True
        return self._state

    async def close(self) -> None:
        """Release resources (SQLite connections, trackers)."""
        db = self._state.get("graph_db")
        if db:
            with contextlib.suppress(Exception):
                await db.close()

        analytics = self._state.get("analytics")
        if analytics:
            with contextlib.suppress(Exception):
                analytics.close()

        feedback = self._state.get("feedback")
        if feedback:
            with contextlib.suppress(Exception):
                feedback.close()

        self._state.clear()
        self._initialized = False


class WorkspaceManager:
    """Routes requests to the correct project and manages project lifecycles.

    Uses an LRU cache to keep at most `max_loaded` projects in memory.
    When the limit is exceeded, the least recently used project is evicted
    (its SQLite connections are closed).
    """

    def __init__(
        self,
        registry: GlobalRegistry | None = None,
        max_loaded: int = 5,
    ) -> None:
        self._registry = registry or GlobalRegistry()
        self._max_loaded = max_loaded
        # OrderedDict for LRU: most recently used at the end
        self._loaded: OrderedDict[str, ProjectState] = OrderedDict()

    @property
    def registry(self) -> GlobalRegistry:
        return self._registry

    @property
    def loaded_projects(self) -> list[str]:
        """Project IDs currently loaded in memory."""
        return list(self._loaded.keys())

    async def resolve(
        self,
        file_path: str | None = None,
        project_id: str | None = None,
    ) -> dict[str, Any]:
        """Resolve and return the initialized state for a project.

        Resolution order:
        1. Explicit project_id → look up in registry
        2. file_path → find which registered project contains it
        3. file_path → walk up to discover unregistered project root
        4. None → error

        Returns the project's state dict (same shape as server.py _state).
        """
        entry: ProjectEntry | None = None

        # 1. Explicit project_id
        if project_id:
            entry = self._registry.get(project_id)
            if not entry:
                return {"error": f"Project '{project_id}' not found in registry"}

        # 2. file_path → match against registered projects
        if not entry and file_path:
            resolved_path = Path(file_path).resolve()  # noqa: ASYNC240
            entry = self._registry.find_by_path(resolved_path)

            # 3. Auto-discover: walk up to find .git or .codebase-intel
            if not entry:
                discovered_root = self._discover_project_root(resolved_path)
                if discovered_root:
                    entry = self._registry.register(discovered_root)
                    logger.info(
                        "Auto-discovered and registered project '%s' at %s",
                        entry.project_id,
                        discovered_root,
                    )

        if not entry:
            return {
                "error": "Cannot determine project. Provide a file_path or project_id.",
                "suggestion": "Register projects with `codebase-intel register <path>`",
            }

        if not entry.is_valid:
            return {
                "error": f"Project root no longer exists: {entry.root_path}",
                "suggestion": f"Run `codebase-intel unregister {entry.project_id}`",
            }

        # Get or create ProjectState
        project_state = await self._get_or_load(entry)
        state = await project_state.ensure_initialized()

        # Inject project metadata for tools to use
        state["_project_id"] = entry.project_id
        state["_project_name"] = entry.name

        return state

    async def _get_or_load(self, entry: ProjectEntry) -> ProjectState:
        """Get a loaded ProjectState or load it, evicting LRU if needed."""
        pid = entry.project_id

        if pid in self._loaded:
            # Move to end (most recently used)
            self._loaded.move_to_end(pid)
            return self._loaded[pid]

        # Evict LRU if at capacity
        if len(self._loaded) >= self._max_loaded:
            await self._evict_lru()

        # Load new project
        project_state = ProjectState(entry.root)
        self._loaded[pid] = project_state
        return project_state

    async def _evict_lru(self) -> None:
        """Evict the least recently used project to free resources."""
        if not self._loaded:
            return

        # OrderedDict: first item is the LRU
        pid, state = self._loaded.popitem(last=False)
        logger.info("Evicting project '%s' from workspace cache", pid)
        await state.close()

    def _discover_project_root(self, path: Path) -> Path | None:
        """Walk up from a file path to find a project root.

        A project root is identified by:
        1. .codebase-intel/ directory (already initialized)
        2. .git/ directory (standard project root)
        3. pyproject.toml, package.json, Cargo.toml, go.mod (build file)

        Stops at filesystem root or home directory.
        """
        root_markers = {
            ".codebase-intel",
            ".git",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "pom.xml",
            "build.gradle",
            "Makefile",
        }

        current = path if path.is_dir() else path.parent
        home = Path.home()

        while current != current.parent:
            # Don't go above home directory
            if current == home.parent:
                break

            for marker in root_markers:
                if (current / marker).exists():
                    return current

            current = current.parent

        return None

    async def close_all(self) -> None:
        """Close all loaded projects. Call on server shutdown."""
        for pid, state in list(self._loaded.items()):
            logger.info("Closing project '%s'", pid)
            await state.close()
        self._loaded.clear()
