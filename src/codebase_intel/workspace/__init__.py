"""Workspace module — global project registry and multi-project routing.

Enables a single MCP server to serve multiple projects by:
1. Maintaining a global registry of known projects (~/.codebase-intel/registry.yaml)
2. Resolving which project a request belongs to based on file paths
3. Lazy-loading per-project state with LRU eviction
4. Auto-discovering project roots by walking up to find .git or .codebase-intel
"""

from codebase_intel.workspace.manager import ProjectState, WorkspaceManager
from codebase_intel.workspace.registry import GlobalRegistry

__all__ = ["GlobalRegistry", "ProjectState", "WorkspaceManager"]
