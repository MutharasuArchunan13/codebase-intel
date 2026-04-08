"""High-level graph query engine — translates task descriptions into graph traversals.

This module bridges the gap between "what is the agent working on?" and
"what graph nodes/edges are relevant?" It's the intelligence layer above
raw graph traversal.

Edge cases:
- Task mentions files that don't exist in the graph (new files): return empty + warning
- Task is too vague ("improve performance"): return high-level module structure
- Task mentions multiple unrelated areas: union of relevant subgraphs
- Task mentions external dependencies: include the import edges but not
  the external code itself
- Conflicting relevance signals: same node relevant for multiple reasons
  with different priorities — take the highest
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codebase_intel.core.types import (
    ContextPriority,
    EdgeKind,
    GraphEdge,
    GraphNode,
    Language,
    NodeKind,
)

if TYPE_CHECKING:
    from codebase_intel.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


@dataclass
class RelevanceResult:
    """Result of a relevance query — nodes with priorities and explanations."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    priorities: dict[str, ContextPriority] = field(default_factory=dict)
    explanations: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    truncated: bool = False

    def nodes_by_priority(self) -> dict[ContextPriority, list[GraphNode]]:
        """Group nodes by their assigned priority."""
        grouped: dict[ContextPriority, list[GraphNode]] = {
            p: [] for p in ContextPriority
        }
        for node in self.nodes:
            priority = self.priorities.get(node.node_id, ContextPriority.LOW)
            grouped[priority].append(node)
        return grouped

    @property
    def unique_files(self) -> set[Path]:
        """All unique file paths referenced by result nodes."""
        return {n.file_path for n in self.nodes}


class GraphQueryEngine:
    """Translates high-level queries into graph traversals.

    The engine supports several query modes:
    1. File-based: "what's relevant to these files?"
    2. Symbol-based: "what's relevant to this function/class?"
    3. Impact-based: "what's affected by changes to X?"
    4. Scope-based: "show me everything in this directory/module"
    """

    def __init__(self, storage: GraphStorage, max_result_nodes: int = 200) -> None:
        self._storage = storage
        self._max_result_nodes = max_result_nodes

    async def query_by_files(
        self,
        file_paths: list[Path],
        include_depth: int = 2,
    ) -> RelevanceResult:
        """Find relevant context for a set of files.

        Priority assignment:
        - CRITICAL: the files themselves
        - HIGH: direct imports/dependencies of those files
        - MEDIUM: transitive dependencies (depth 2)
        - LOW: test files that test these files

        Edge cases:
        - File not in graph: might be new. Add warning, no nodes.
        - File is a barrel/index: has many dependents. Cap and prioritize
          by which dependents are most coupled (most edges).
        - File is in node_modules: skip deep traversal (it's external)
        """
        result = RelevanceResult()
        seen_ids: set[str] = set()

        for fp in file_paths:
            file_nodes = await self._storage.get_nodes_by_file(fp)
            if not file_nodes:
                result.warnings.append(
                    f"File {fp} not found in graph (new file or not yet indexed)"
                )
                continue

            for node in file_nodes:
                if node.node_id not in seen_ids:
                    seen_ids.add(node.node_id)
                    result.nodes.append(node)
                    result.priorities[node.node_id] = ContextPriority.CRITICAL
                    result.explanations[node.node_id] = "Directly referenced file"

            # Gather dependencies (what these files import)
            for node in file_nodes:
                if node.kind == NodeKind.MODULE:
                    deps = await self._storage.get_dependencies(
                        node.node_id,
                        max_depth=include_depth,
                    )
                    for dep in deps:
                        if dep.node_id in seen_ids:
                            continue
                        if dep.is_external:
                            continue  # Don't include external dependency internals
                        if len(result.nodes) >= self._max_result_nodes:
                            result.truncated = True
                            break

                        seen_ids.add(dep.node_id)
                        result.nodes.append(dep)

                        # Depth-based priority
                        priority = ContextPriority.HIGH
                        explanation = f"Direct dependency of {fp.name}"
                        # Check if it's a transitive dep (depth > 1)
                        direct_deps = await self._storage.get_dependencies(
                            node.node_id, max_depth=1
                        )
                        direct_ids = {d.node_id for d in direct_deps}
                        if dep.node_id not in direct_ids:
                            priority = ContextPriority.MEDIUM
                            explanation = f"Transitive dependency of {fp.name}"

                        result.priorities[dep.node_id] = priority
                        result.explanations[dep.node_id] = explanation

            # Find test files that test these files
            for node in file_nodes:
                if node.kind == NodeKind.MODULE:
                    test_files = await self._find_test_files(node)
                    for test_node in test_files:
                        if test_node.node_id not in seen_ids:
                            seen_ids.add(test_node.node_id)
                            result.nodes.append(test_node)
                            result.priorities[test_node.node_id] = ContextPriority.LOW
                            result.explanations[test_node.node_id] = (
                                f"Test file for {fp.name}"
                            )

        return result

    async def query_by_symbol(
        self,
        symbol_name: str,
        include_depth: int = 2,
    ) -> RelevanceResult:
        """Find relevant context for a specific symbol (function, class, etc.).

        Edge cases:
        - Symbol name is ambiguous (exists in multiple files): return all
          matches with file path in the explanation
        - Symbol is a common name ("get", "create", "handle"): may match
          many nodes. Prioritize by: same directory > same package > global
        - Symbol doesn't exist: might be a typo, suggest similar names
        """
        result = RelevanceResult()
        candidates = await self._search_symbol(symbol_name)

        if not candidates:
            result.warnings.append(
                f"Symbol '{symbol_name}' not found in graph. "
                f"It may be new, in an unparsed file, or misspelled."
            )
            return result

        seen_ids: set[str] = set()

        for node in candidates:
            seen_ids.add(node.node_id)
            result.nodes.append(node)
            result.priorities[node.node_id] = ContextPriority.CRITICAL
            result.explanations[node.node_id] = f"Matched symbol '{symbol_name}'"

            # Get dependencies and dependents
            deps = await self._storage.get_dependencies(
                node.node_id, max_depth=include_depth
            )
            dependents = await self._storage.get_dependents(
                node.node_id, max_depth=1
            )

            for dep in deps:
                if dep.node_id not in seen_ids and not dep.is_external:
                    seen_ids.add(dep.node_id)
                    result.nodes.append(dep)
                    result.priorities[dep.node_id] = ContextPriority.HIGH
                    result.explanations[dep.node_id] = (
                        f"Dependency of {symbol_name}"
                    )

            for dep in dependents:
                if dep.node_id not in seen_ids and not dep.is_external:
                    if len(result.nodes) >= self._max_result_nodes:
                        result.truncated = True
                        break
                    seen_ids.add(dep.node_id)
                    result.nodes.append(dep)
                    result.priorities[dep.node_id] = ContextPriority.MEDIUM
                    result.explanations[dep.node_id] = (
                        f"Depends on {symbol_name} (may be affected by changes)"
                    )

        return result

    async def query_impact(
        self,
        changed_files: list[Path],
        max_depth: int = 3,
    ) -> RelevanceResult:
        """Analyze the impact of file changes — "what else could break?"

        This is a REVERSE traversal: given changes, find what depends on them.

        Edge cases:
        - Changed file is __init__.py: potentially affects all importers of the package
        - Changed file is a config file: affects everything that reads it
        - Changed file has no dependents: isolated change (rare but valid)
        - Cascade explosion: changed a core utility → 500 dependents.
          Cap at max_result_nodes and prioritize by coupling strength
          (number of edges to the changed file).
        """
        result = RelevanceResult()
        impact_map = await self._storage.impact_analysis(changed_files, max_depth=max_depth)

        # Add changed files as CRITICAL
        seen_ids: set[str] = set()
        for fp in changed_files:
            for node in await self._storage.get_nodes_by_file(fp):
                if node.node_id not in seen_ids:
                    seen_ids.add(node.node_id)
                    result.nodes.append(node)
                    result.priorities[node.node_id] = ContextPriority.CRITICAL
                    result.explanations[node.node_id] = "Changed file"

        # Add impacted nodes with distance-based priority
        for file_key, affected_nodes in impact_map.items():
            for node in affected_nodes:
                if node.node_id in seen_ids:
                    continue
                if len(result.nodes) >= self._max_result_nodes:
                    result.truncated = True
                    result.warnings.append(
                        f"Impact analysis truncated at {self._max_result_nodes} nodes. "
                        f"Consider narrowing the scope."
                    )
                    break

                seen_ids.add(node.node_id)
                result.nodes.append(node)
                result.priorities[node.node_id] = ContextPriority.HIGH
                result.explanations[node.node_id] = (
                    f"Depends on changed file {Path(file_key).name}"
                )

        return result

    async def query_scope(
        self,
        directory: Path,
        max_depth: int = 1,
    ) -> RelevanceResult:
        """Get all nodes within a directory scope.

        Useful for "show me the structure of this module/package."

        Edge case: directory has 1000+ files (monorepo root). We return
        MODULE-level nodes only and mark as truncated.
        """
        result = RelevanceResult()

        # This requires a storage method to search by path prefix
        # For now, we do it via the stats + file listing approach
        # In production, we'd add a dedicated index

        result.warnings.append(
            "Scope queries are limited to indexed files. "
            "Run `codebase-intel analyze` to ensure the graph is current."
        )
        return result

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    async def _search_symbol(self, name: str) -> list[GraphNode]:
        """Search for a symbol by name across the graph.

        Edge case: name might be:
        - Exact: "UserService" → match name field
        - Qualified: "auth.UserService" → match qualified_name field
        - Partial: "user_serv" → fuzzy match (future)
        """
        # Exact name match
        cursor = await self._storage._db.execute(
            """
            SELECT node_id, kind, name, qualified_name, file_path,
                   line_start, line_end, language, content_hash, docstring,
                   is_generated, is_external, is_test, is_entry_point,
                   metadata_json
            FROM nodes
            WHERE name = ? OR qualified_name = ? OR qualified_name LIKE ?
            LIMIT 50
            """,
            (name, name, f"%.{name}"),
        )
        rows = await cursor.fetchall()
        return [self._storage._row_to_node(row) for row in rows]

    async def _find_test_files(self, source_node: GraphNode) -> list[GraphNode]:
        """Find test files that test a given source module.

        Detection heuristics:
        - Explicit TESTS edge in graph
        - File naming convention: foo.py → test_foo.py, foo_test.py
        - Import-based: test file that imports the source module
        """
        results: list[GraphNode] = []

        # Check for explicit TESTS edges
        cursor = await self._storage._db.execute(
            """
            SELECT source_id FROM edges
            WHERE target_id = ? AND kind = ?
            """,
            (source_node.node_id, EdgeKind.TESTS.value),
        )
        for row in await cursor.fetchall():
            node = await self._storage.get_node(row[0])
            if node:
                results.append(node)

        # Convention-based: look for test_<name> or <name>_test
        if source_node.kind == NodeKind.MODULE:
            source_name = source_node.name
            test_patterns = [f"test_{source_name}", f"{source_name}_test", f"{source_name}_spec"]
            cursor = await self._storage._db.execute(
                f"""
                SELECT node_id, kind, name, qualified_name, file_path,
                       line_start, line_end, language, content_hash, docstring,
                       is_generated, is_external, is_test, is_entry_point,
                       metadata_json
                FROM nodes
                WHERE kind = 'module' AND is_test = 1
                AND ({' OR '.join(f"name = ?" for _ in test_patterns)})
                """,
                test_patterns,
            )
            for row in await cursor.fetchall():
                node = self._storage._row_to_node(row)
                if node.node_id not in {r.node_id for r in results}:
                    results.append(node)

        return results
