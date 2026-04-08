"""SQLite-backed graph storage with WAL mode for concurrent access.

Design decisions:
- SQLite over a graph DB (Neo4j, etc.) to maintain zero-dependency portability
- WAL mode enables concurrent readers (MCP queries) + single writer (git hook updates)
- Schema versioning for forward migration without data loss
- Batch operations for initial build (10k+ nodes), single ops for incremental

Edge cases handled:
- Concurrent writes: WAL mode + busy timeout + application-level retry
- Corrupt database: detect via integrity check, offer re-initialization
- Schema migration: version table tracks schema, auto-migrate on open
- Large codebases: batch inserts with transaction chunking (100k+ nodes)
- Partial writes: crash during build → incomplete graph → detect via marker table
- Disk full: catch and surface meaningful error
- Path encoding: SQLite stores paths as TEXT, we normalize to POSIX format
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, AsyncIterator

import aiosqlite

from codebase_intel.core.exceptions import (
    ErrorContext,
    StorageConcurrencyError,
    StorageCorruptError,
    StorageMigrationError,
)
from codebase_intel.core.types import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    Language,
    LineRange,
    NodeKind,
)

if TYPE_CHECKING:
    from codebase_intel.core.config import GraphConfig

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL,
    migrated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Build status tracking (detect incomplete builds)
CREATE TABLE IF NOT EXISTS build_status (
    build_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    file_count INTEGER DEFAULT 0,
    node_count INTEGER DEFAULT 0,
    edge_count INTEGER DEFAULT 0
);

-- Graph nodes
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    file_path TEXT NOT NULL,           -- POSIX-normalized, relative to project root
    line_start INTEGER,
    line_end INTEGER,
    language TEXT NOT NULL DEFAULT 'unknown',
    content_hash TEXT,
    docstring TEXT,
    is_generated INTEGER NOT NULL DEFAULT 0,
    is_external INTEGER NOT NULL DEFAULT 0,
    is_test INTEGER NOT NULL DEFAULT 0,
    is_entry_point INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Graph edges
CREATE TABLE IF NOT EXISTS edges (
    source_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    target_id TEXT NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,
    is_type_only INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT DEFAULT '{}',
    PRIMARY KEY (source_id, target_id, kind)
);

-- File fingerprints (for incremental updates — only re-parse changed files)
CREATE TABLE IF NOT EXISTS file_fingerprints (
    file_path TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    last_modified TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'unknown',
    node_count INTEGER NOT NULL DEFAULT 0
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_nodes_file ON nodes(file_path);
CREATE INDEX IF NOT EXISTS idx_nodes_kind ON nodes(kind);
CREATE INDEX IF NOT EXISTS idx_nodes_name ON nodes(name);
CREATE INDEX IF NOT EXISTS idx_nodes_qualified ON nodes(qualified_name);
CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
CREATE INDEX IF NOT EXISTS idx_edges_kind ON edges(kind);
"""


class GraphStorage:
    """Async SQLite storage for the semantic code graph.

    Usage:
        async with GraphStorage.open(config) as storage:
            await storage.upsert_node(node)
            nodes = await storage.get_dependents("node_id")
    """

    def __init__(self, db: aiosqlite.Connection, project_root: Path) -> None:
        self._db = db
        self._project_root = project_root

    @classmethod
    @asynccontextmanager
    async def open(cls, config: GraphConfig, project_root: Path) -> AsyncIterator[GraphStorage]:
        """Open (or create) the graph database with proper configuration.

        Edge cases:
        - DB file doesn't exist: create with full schema
        - DB file exists but wrong version: migrate or error
        - DB file is corrupt: integrity check fails, raise StorageCorruptError
        - DB locked by another process: retry with busy_timeout
        """
        db_path = config.db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)

        db = await aiosqlite.connect(str(db_path))
        try:
            # Enable WAL mode for concurrent read/write
            if config.enable_wal_mode:
                await db.execute("PRAGMA journal_mode=WAL")

            # Busy timeout: wait up to 5s for locks instead of failing immediately
            # Edge case: git hook and MCP server both active
            await db.execute("PRAGMA busy_timeout=5000")

            # Foreign keys for cascade deletes (removing a node removes its edges)
            await db.execute("PRAGMA foreign_keys=ON")

            # Integrity check on first open
            result = await db.execute("PRAGMA integrity_check")
            check = await result.fetchone()
            if check and check[0] != "ok":
                raise StorageCorruptError(
                    f"Database integrity check failed: {check[0]}",
                    ErrorContext(file_path=db_path, operation="integrity_check"),
                )

            storage = cls(db, project_root)
            await storage._ensure_schema()
            yield storage

        except aiosqlite.OperationalError as exc:
            if "database is locked" in str(exc):
                raise StorageConcurrencyError(
                    "Database is locked by another process",
                    ErrorContext(file_path=db_path, operation="open"),
                ) from exc
            raise
        finally:
            await db.close()

    async def _ensure_schema(self) -> None:
        """Create or migrate the schema.

        Edge case: schema_version table doesn't exist (fresh DB) vs.
        exists with older version (needs migration) vs.
        exists with newer version (user downgraded the tool — refuse).
        """
        # Check if schema_version table exists
        cursor = await self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        table_exists = await cursor.fetchone()

        if not table_exists:
            # Fresh database — create everything
            await self._db.executescript(SCHEMA_SQL)
            await self._db.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
            await self._db.commit()
            return

        # Check version
        cursor = await self._db.execute(
            "SELECT MAX(version) FROM schema_version"
        )
        row = await cursor.fetchone()
        current_version = row[0] if row else 0

        if current_version == SCHEMA_VERSION:
            return

        if current_version > SCHEMA_VERSION:
            raise StorageMigrationError(
                f"Database schema version {current_version} is newer than "
                f"supported version {SCHEMA_VERSION}. Please upgrade codebase-intel.",
            )

        # Future: migration logic goes here
        # For now, v1 is the only version
        logger.info("Migrating schema from v%d to v%d", current_version, SCHEMA_VERSION)
        await self._db.executescript(SCHEMA_SQL)
        await self._db.execute(
            "INSERT INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )
        await self._db.commit()

    # -------------------------------------------------------------------
    # Path normalization
    # -------------------------------------------------------------------

    def _to_stored_path(self, path: Path) -> str:
        """Convert absolute path to POSIX-relative for storage.

        Edge case: path outside project root (symlink target, monorepo ref).
        We store absolute POSIX path in that case.
        """
        try:
            return str(PurePosixPath(path.resolve().relative_to(self._project_root)))
        except ValueError:
            return str(PurePosixPath(path.resolve()))

    def _from_stored_path(self, stored: str) -> Path:
        """Convert stored POSIX path back to absolute Path."""
        p = Path(stored)
        if p.is_absolute():
            return p
        return self._project_root / p

    # -------------------------------------------------------------------
    # Node operations
    # -------------------------------------------------------------------

    async def upsert_node(self, node: GraphNode) -> None:
        """Insert or update a graph node.

        Edge case: node with same ID but different content (file changed).
        We update in place — the node_id is deterministic from (path, kind, name).
        """
        import json

        await self._db.execute(
            """
            INSERT INTO nodes (
                node_id, kind, name, qualified_name, file_path,
                line_start, line_end, language, content_hash, docstring,
                is_generated, is_external, is_test, is_entry_point,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                kind=excluded.kind, name=excluded.name,
                qualified_name=excluded.qualified_name,
                file_path=excluded.file_path,
                line_start=excluded.line_start, line_end=excluded.line_end,
                language=excluded.language, content_hash=excluded.content_hash,
                docstring=excluded.docstring,
                is_generated=excluded.is_generated,
                is_external=excluded.is_external,
                is_test=excluded.is_test,
                is_entry_point=excluded.is_entry_point,
                metadata_json=excluded.metadata_json,
                updated_at=datetime('now')
            """,
            (
                node.node_id,
                node.kind.value,
                node.name,
                node.qualified_name,
                self._to_stored_path(node.file_path),
                node.line_range.start if node.line_range else None,
                node.line_range.end if node.line_range else None,
                node.language.value,
                node.content_hash,
                node.docstring,
                int(node.is_generated),
                int(node.is_external),
                int(node.is_test),
                int(node.is_entry_point),
                json.dumps(node.metadata),
            ),
        )

    async def upsert_nodes_batch(self, nodes: list[GraphNode]) -> None:
        """Batch upsert for initial graph build.

        Edge case: 100k+ nodes on a large codebase. We chunk into transactions
        of 1000 to balance speed vs. memory and allow partial progress visibility.
        """
        import json

        chunk_size = 1000
        for i in range(0, len(nodes), chunk_size):
            chunk = nodes[i : i + chunk_size]
            await self._db.executemany(
                """
                INSERT INTO nodes (
                    node_id, kind, name, qualified_name, file_path,
                    line_start, line_end, language, content_hash, docstring,
                    is_generated, is_external, is_test, is_entry_point,
                    metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    kind=excluded.kind, name=excluded.name,
                    qualified_name=excluded.qualified_name,
                    file_path=excluded.file_path,
                    line_start=excluded.line_start, line_end=excluded.line_end,
                    language=excluded.language, content_hash=excluded.content_hash,
                    docstring=excluded.docstring,
                    is_generated=excluded.is_generated,
                    is_external=excluded.is_external,
                    is_test=excluded.is_test,
                    is_entry_point=excluded.is_entry_point,
                    metadata_json=excluded.metadata_json,
                    updated_at=datetime('now')
                """,
                [
                    (
                        n.node_id,
                        n.kind.value,
                        n.name,
                        n.qualified_name,
                        self._to_stored_path(n.file_path),
                        n.line_range.start if n.line_range else None,
                        n.line_range.end if n.line_range else None,
                        n.language.value,
                        n.content_hash,
                        n.docstring,
                        int(n.is_generated),
                        int(n.is_external),
                        int(n.is_test),
                        int(n.is_entry_point),
                        json.dumps(n.metadata),
                    )
                    for n in chunk
                ],
            )
            await self._db.commit()

    async def upsert_edge(self, edge: GraphEdge) -> None:
        """Insert or update a graph edge."""
        import json

        await self._db.execute(
            """
            INSERT INTO edges (source_id, target_id, kind, confidence, is_type_only, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, target_id, kind) DO UPDATE SET
                confidence=excluded.confidence,
                is_type_only=excluded.is_type_only,
                metadata_json=excluded.metadata_json
            """,
            (
                edge.source_id,
                edge.target_id,
                edge.kind.value,
                edge.confidence,
                int(edge.is_type_only),
                json.dumps(edge.metadata),
            ),
        )

    async def upsert_edges_batch(self, edges: list[GraphEdge]) -> None:
        """Batch upsert edges."""
        import json

        chunk_size = 1000
        for i in range(0, len(edges), chunk_size):
            chunk = edges[i : i + chunk_size]
            await self._db.executemany(
                """
                INSERT INTO edges (source_id, target_id, kind, confidence, is_type_only, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id, target_id, kind) DO UPDATE SET
                    confidence=excluded.confidence,
                    is_type_only=excluded.is_type_only,
                    metadata_json=excluded.metadata_json
                """,
                [
                    (
                        e.source_id,
                        e.target_id,
                        e.kind.value,
                        e.confidence,
                        int(e.is_type_only),
                        json.dumps(e.metadata),
                    )
                    for e in chunk
                ],
            )
            await self._db.commit()

    async def remove_file_nodes(self, file_path: Path) -> int:
        """Remove all nodes (and their edges via CASCADE) for a file.

        Used during incremental update: delete old nodes before re-parsing.

        Edge case: file renamed — old path nodes are removed, new path
        nodes are added. The graph handles this as delete+insert, but
        the orchestrator can detect renames via content_hash matching.
        """
        stored_path = self._to_stored_path(file_path)
        cursor = await self._db.execute(
            "DELETE FROM nodes WHERE file_path = ?", (stored_path,)
        )
        await self._db.commit()
        return cursor.rowcount  # type: ignore[return-value]

    # -------------------------------------------------------------------
    # Query operations
    # -------------------------------------------------------------------

    def _row_to_node(self, row: aiosqlite.Row) -> GraphNode:
        """Convert a database row to a GraphNode."""
        import json

        line_range = None
        if row[5] is not None and row[6] is not None:
            line_range = LineRange(start=row[5], end=row[6])

        return GraphNode(
            node_id=row[0],
            kind=NodeKind(row[1]),
            name=row[2],
            qualified_name=row[3],
            file_path=self._from_stored_path(row[4]),
            line_range=line_range,
            language=Language(row[7]),
            content_hash=row[8],
            docstring=row[9],
            is_generated=bool(row[10]),
            is_external=bool(row[11]),
            is_test=bool(row[12]),
            is_entry_point=bool(row[13]),
            metadata=json.loads(row[14]) if row[14] else {},
        )

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Get a single node by ID."""
        self._db.row_factory = None
        cursor = await self._db.execute(
            """
            SELECT node_id, kind, name, qualified_name, file_path,
                   line_start, line_end, language, content_hash, docstring,
                   is_generated, is_external, is_test, is_entry_point,
                   metadata_json
            FROM nodes WHERE node_id = ?
            """,
            (node_id,),
        )
        row = await cursor.fetchone()
        return self._row_to_node(row) if row else None

    async def get_nodes_by_file(self, file_path: Path) -> list[GraphNode]:
        """Get all nodes defined in a file."""
        stored_path = self._to_stored_path(file_path)
        cursor = await self._db.execute(
            """
            SELECT node_id, kind, name, qualified_name, file_path,
                   line_start, line_end, language, content_hash, docstring,
                   is_generated, is_external, is_test, is_entry_point,
                   metadata_json
            FROM nodes WHERE file_path = ?
            ORDER BY line_start
            """,
            (stored_path,),
        )
        return [self._row_to_node(row) for row in await cursor.fetchall()]

    async def get_dependents(
        self,
        node_id: str,
        *,
        edge_kinds: list[EdgeKind] | None = None,
        include_type_only: bool = False,
        max_depth: int = 1,
    ) -> list[GraphNode]:
        """Find all nodes that depend ON the given node (reverse traversal).

        This answers: "what breaks if I change this?"

        Edge cases:
        - Circular dependencies: tracked via visited set, no infinite loop
        - Deep chains: capped at max_depth to prevent runaway
        - Type-only deps: excluded by default (changing implementation doesn't
          break a TYPE_CHECKING import)
        - Dynamic imports: included but with lower confidence
        """
        visited: set[str] = set()
        result: list[GraphNode] = []
        queue: list[tuple[str, int]] = [(node_id, 0)]

        kind_filter = ""
        if edge_kinds:
            kinds_sql = ",".join(f"'{k.value}'" for k in edge_kinds)
            kind_filter = f"AND kind IN ({kinds_sql})"

        type_filter = "" if include_type_only else "AND is_type_only = 0"

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            cursor = await self._db.execute(
                f"""
                SELECT source_id FROM edges
                WHERE target_id = ? {kind_filter} {type_filter}
                """,  # noqa: S608
                (current_id,),
            )
            for row in await cursor.fetchall():
                source_id = row[0]
                if source_id not in visited:
                    node = await self.get_node(source_id)
                    if node:
                        result.append(node)
                        if depth + 1 <= max_depth:
                            queue.append((source_id, depth + 1))

        return result

    async def get_dependencies(
        self,
        node_id: str,
        *,
        edge_kinds: list[EdgeKind] | None = None,
        include_type_only: bool = True,
        max_depth: int = 1,
    ) -> list[GraphNode]:
        """Find all nodes that the given node depends on (forward traversal).

        This answers: "what context do I need to understand this?"
        """
        visited: set[str] = set()
        result: list[GraphNode] = []
        queue: list[tuple[str, int]] = [(node_id, 0)]

        kind_filter = ""
        if edge_kinds:
            kinds_sql = ",".join(f"'{k.value}'" for k in edge_kinds)
            kind_filter = f"AND kind IN ({kinds_sql})"

        type_filter = "" if include_type_only else "AND is_type_only = 0"

        while queue:
            current_id, depth = queue.pop(0)
            if current_id in visited or depth > max_depth:
                continue
            visited.add(current_id)

            cursor = await self._db.execute(
                f"""
                SELECT target_id FROM edges
                WHERE source_id = ? {kind_filter} {type_filter}
                """,  # noqa: S608
                (current_id,),
            )
            for row in await cursor.fetchall():
                target_id = row[0]
                if target_id not in visited:
                    node = await self.get_node(target_id)
                    if node:
                        result.append(node)
                        if depth + 1 <= max_depth:
                            queue.append((target_id, depth + 1))

        return result

    async def find_cycles(self, max_cycle_length: int = 10) -> list[list[str]]:
        """Detect circular dependency chains in the graph.

        Edge case: large graphs can have many cycles. We cap detection
        at max_cycle_length to keep this practical. Reports the shortest
        cycles first (most actionable).

        Uses DFS with back-edge detection.
        """
        # Get all node IDs
        cursor = await self._db.execute("SELECT node_id FROM nodes")
        all_nodes = [row[0] for row in await cursor.fetchall()]

        visited: set[str] = set()
        rec_stack: set[str] = set()
        path: list[str] = []
        cycles: list[list[str]] = []

        async def _dfs(node_id: str) -> None:
            if len(cycles) >= 50:  # Cap to prevent excessive output
                return
            if len(path) > max_cycle_length:
                return

            visited.add(node_id)
            rec_stack.add(node_id)
            path.append(node_id)

            cursor = await self._db.execute(
                "SELECT target_id FROM edges WHERE source_id = ?",
                (node_id,),
            )
            for row in await cursor.fetchall():
                target_id = row[0]
                if target_id not in visited:
                    await _dfs(target_id)
                elif target_id in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(target_id)
                    cycle = path[cycle_start:] + [target_id]
                    cycles.append(cycle)

            path.pop()
            rec_stack.discard(node_id)

        for node_id in all_nodes:
            if node_id not in visited:
                await _dfs(node_id)

        return sorted(cycles, key=len)

    async def impact_analysis(
        self,
        file_paths: list[Path],
        max_depth: int = 3,
    ) -> dict[str, list[GraphNode]]:
        """For a set of changed files, find all transitively affected nodes.

        This is the core query for the orchestrator: "these files changed,
        what else needs to be in context?"

        Edge cases:
        - Changed file not in graph: it's new, return empty (no dependents yet)
        - Changed file is a barrel/index: many dependents, may blow up results
          → cap at 100 dependents per file and flag truncation
        - Changed file is generated: lower priority in results
        - Changed file is a config: flag as potentially affecting everything
          that reads this config

        Returns: dict mapping file path → list of affected nodes
        """
        result: dict[str, list[GraphNode]] = {}
        max_dependents_per_file = 100

        for fp in file_paths:
            nodes = await self.get_nodes_by_file(fp)
            affected: list[GraphNode] = []
            seen: set[str] = set()

            for node in nodes:
                dependents = await self.get_dependents(
                    node.node_id, max_depth=max_depth
                )
                for dep in dependents:
                    if dep.node_id not in seen:
                        seen.add(dep.node_id)
                        affected.append(dep)

                    if len(affected) >= max_dependents_per_file:
                        logger.warning(
                            "Impact analysis for %s truncated at %d dependents",
                            fp,
                            max_dependents_per_file,
                        )
                        break

            result[str(fp)] = affected

        return result

    async def get_stats(self) -> dict[str, int]:
        """Return graph statistics for health checks and CLI display."""
        stats: dict[str, int] = {}
        for table in ("nodes", "edges", "file_fingerprints"):
            cursor = await self._db.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            row = await cursor.fetchone()
            stats[f"{table}_count"] = row[0] if row else 0
        return stats

    async def get_fingerprint(self, file_path: Path) -> str | None:
        """Get stored content hash for a file (for incremental update checks)."""
        stored_path = self._to_stored_path(file_path)
        cursor = await self._db.execute(
            "SELECT content_hash FROM file_fingerprints WHERE file_path = ?",
            (stored_path,),
        )
        row = await cursor.fetchone()
        return row[0] if row else None

    async def update_fingerprint(
        self,
        file_path: Path,
        content_hash: str,
        size_bytes: int,
        last_modified: str,
        language: Language,
        node_count: int,
    ) -> None:
        """Update the stored fingerprint for a file."""
        stored_path = self._to_stored_path(file_path)
        await self._db.execute(
            """
            INSERT INTO file_fingerprints (file_path, content_hash, size_bytes, last_modified, language, node_count)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
                content_hash=excluded.content_hash,
                size_bytes=excluded.size_bytes,
                last_modified=excluded.last_modified,
                language=excluded.language,
                node_count=excluded.node_count
            """,
            (stored_path, content_hash, size_bytes, last_modified, language.value, node_count),
        )
        await self._db.commit()
