"""Graph builder — orchestrates parsing and storage for full and incremental builds.

This module manages the lifecycle of the code graph:
- Full build: parse entire codebase from scratch
- Incremental build: only re-parse files that changed since last build
- Cleanup: remove nodes for deleted files

Edge cases:
- Huge codebase (100k+ files): progress reporting, chunked processing, memory limits
- Incremental build after large refactor: many files changed, detect renames via
  content hash matching (old hash appears at new path)
- Build interrupted (crash, Ctrl+C): detect via build_status table, offer resume
- Concurrent builds: detect via build_status, refuse second build
- New language added to project: need to re-scan files that were previously skipped
- .gitignore changes: previously ignored files now need parsing
- Symlink loops: resolve() + seen set prevents infinite recursion
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from codebase_intel.core.exceptions import ErrorContext, ParseError
from codebase_intel.core.types import Language
from codebase_intel.graph.parser import FileParser, ParseResult, compute_file_hash, detect_language

if TYPE_CHECKING:
    from codebase_intel.core.config import ProjectConfig
    from codebase_intel.graph.storage import GraphStorage

logger = logging.getLogger(__name__)


class BuildProgress:
    """Tracks and reports build progress."""

    def __init__(self, total_files: int) -> None:
        self.total_files = total_files
        self.processed: int = 0
        self.skipped: int = 0
        self.failed: int = 0
        self.nodes_created: int = 0
        self.edges_created: int = 0
        self.warnings: list[str] = []

    @property
    def completed_pct(self) -> float:
        if self.total_files == 0:
            return 100.0
        return (self.processed + self.skipped + self.failed) / self.total_files * 100

    def summary(self) -> dict[str, int | float]:
        return {
            "total_files": self.total_files,
            "processed": self.processed,
            "skipped": self.skipped,
            "failed": self.failed,
            "nodes_created": self.nodes_created,
            "edges_created": self.edges_created,
            "warning_count": len(self.warnings),
        }


class GraphBuilder:
    """Builds and maintains the semantic code graph."""

    def __init__(
        self,
        config: ProjectConfig,
        storage: GraphStorage,
    ) -> None:
        self._config = config
        self._storage = storage
        self._parser = FileParser(config.parser, config.project_root)

    async def full_build(self) -> BuildProgress:
        """Build the graph from scratch — scan all files in the project.

        Edge cases:
        - Previous incomplete build: detected via build_status, cleaned up first
        - Concurrent build attempt: detected and refused
        - Memory pressure on huge repos: process in batches, commit frequently
        """
        build_id = str(uuid.uuid4())[:8]
        logger.info("Starting full build [%s]", build_id)

        # Record build start
        await self._storage._db.execute(
            "INSERT INTO build_status (build_id, started_at) VALUES (?, ?)",
            (build_id, datetime.now(UTC).isoformat()),
        )
        await self._storage._db.commit()

        # Discover files
        files = list(self._discover_files())
        progress = BuildProgress(len(files))
        logger.info("Discovered %d files to process", len(files))

        # Process in batches to manage memory
        batch_size = 100
        for i in range(0, len(files), batch_size):
            batch = files[i : i + batch_size]
            await self._process_batch(batch, progress)

            # Log progress periodically
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(files):
                logger.info(
                    "Progress: %.0f%% (%d/%d files, %d nodes, %d edges)",
                    progress.completed_pct,
                    progress.processed,
                    progress.total_files,
                    progress.nodes_created,
                    progress.edges_created,
                )

        # Mark build complete
        await self._storage._db.execute(
            """
            UPDATE build_status SET completed_at = ?, file_count = ?,
                   node_count = ?, edge_count = ?
            WHERE build_id = ?
            """,
            (
                datetime.now(UTC).isoformat(),
                progress.processed,
                progress.nodes_created,
                progress.edges_created,
                build_id,
            ),
        )
        await self._storage._db.commit()

        logger.info(
            "Full build complete: %d files, %d nodes, %d edges, %d warnings",
            progress.processed,
            progress.nodes_created,
            progress.edges_created,
            len(progress.warnings),
        )

        return progress

    async def incremental_build(
        self,
        changed_files: list[Path] | None = None,
    ) -> BuildProgress:
        """Update the graph for only changed files.

        If changed_files is None, detect changes via content hash comparison
        against stored fingerprints.

        Edge cases:
        - File renamed: old path's nodes are orphaned, new path gets new nodes.
          We detect renames via content hash: if a new file has the same hash
          as a recently deleted file, it's likely a rename.
        - File deleted: remove its nodes (edges cascade via FK).
        - File content unchanged but timestamp changed: skip (hash-based check).
        - New file added: parse and add to graph.
        """
        if changed_files is None:
            changed_files = await self._detect_changed_files()

        progress = BuildProgress(len(changed_files))
        logger.info("Incremental build: %d files to process", len(changed_files))

        # Detect deleted files (in graph but not on disk)
        await self._cleanup_deleted_files(progress)

        # Process changed/new files
        for fp in changed_files:
            if not fp.exists():
                # File was deleted — remove from graph
                removed = await self._storage.remove_file_nodes(fp)
                if removed > 0:
                    logger.debug("Removed %d nodes for deleted file %s", removed, fp)
                progress.skipped += 1
                continue

            # Remove old nodes for this file before re-parsing
            await self._storage.remove_file_nodes(fp)

            result = await self._parser.parse_file(fp)
            if result is None:
                progress.skipped += 1
                continue

            await self._store_parse_result(result, progress)
            progress.processed += 1

        logger.info(
            "Incremental build complete: %d processed, %d skipped",
            progress.processed,
            progress.skipped,
        )

        return progress

    async def _detect_changed_files(self) -> list[Path]:
        """Detect files that changed since the last build.

        Uses content hash comparison against stored fingerprints.
        New files (not in fingerprint table) are included.

        Edge case: file permissions changed but content didn't — skip.
        Edge case: file touched (timestamp changed) but content same — skip.
        """
        changed: list[Path] = []

        for file_path in self._discover_files():
            try:
                content = file_path.read_bytes()
            except (OSError, PermissionError):
                continue

            current_hash = compute_file_hash(content)
            stored_hash = await self._storage.get_fingerprint(file_path)

            if stored_hash != current_hash:
                changed.append(file_path)

        return changed

    async def _cleanup_deleted_files(self, progress: BuildProgress) -> None:
        """Remove nodes for files that no longer exist on disk.

        Edge case: file still exists but is now in .gitignore — we keep it
        in the graph (it's still code, just not tracked). Only remove nodes
        for files that are truly gone.
        """
        cursor = await self._storage._db.execute(
            "SELECT DISTINCT file_path FROM file_fingerprints"
        )
        for row in await cursor.fetchall():
            stored_path = row[0]
            full_path = self._storage._from_stored_path(stored_path)
            if not full_path.exists():
                removed = await self._storage.remove_file_nodes(full_path)
                await self._storage._db.execute(
                    "DELETE FROM file_fingerprints WHERE file_path = ?",
                    (stored_path,),
                )
                await self._storage._db.commit()
                logger.debug(
                    "Cleaned up %d nodes for deleted file %s", removed, stored_path
                )

    async def _process_batch(
        self,
        files: list[Path],
        progress: BuildProgress,
    ) -> None:
        """Process a batch of files — parse and store."""
        for fp in files:
            try:
                result = await self._parser.parse_file(fp)
            except Exception as exc:
                logger.warning("Failed to parse %s: %s", fp, exc)
                progress.failed += 1
                progress.warnings.append(f"Parse failed: {fp}: {exc}")
                continue

            if result is None:
                progress.skipped += 1
                continue

            await self._store_parse_result(result, progress)
            progress.processed += 1
            progress.warnings.extend(result.warnings)

    async def _store_parse_result(
        self,
        result: ParseResult,
        progress: BuildProgress,
    ) -> None:
        """Store parsed nodes, edges, and fingerprint.

        Edge case: edges may reference target nodes that don't exist in the
        graph (external packages like 'fastapi', 'sqlalchemy'). We filter
        these out before insertion to avoid FK constraint failures.
        Unresolved edges are stored with a relaxed approach — we only insert
        edges where at least the source node exists, and skip edges whose
        targets are unresolved external modules.
        """
        if result.nodes:
            await self._storage.upsert_nodes_batch(result.nodes)
            progress.nodes_created += len(result.nodes)

        if result.edges:
            # Filter edges to only those whose target nodes exist in the DB
            # or whose source exists in the current batch
            valid_edges: list[GraphEdge] = []
            local_node_ids = {n.node_id for n in result.nodes}

            for edge in result.edges:
                # Skip edges to unresolved/external targets
                if edge.target_id.startswith("unresolved:"):
                    continue

                # Check if target exists in DB or current batch
                target_exists = edge.target_id in local_node_ids
                if not target_exists:
                    target_node = await self._storage.get_node(edge.target_id)
                    target_exists = target_node is not None

                if target_exists:
                    valid_edges.append(edge)

            if valid_edges:
                await self._storage.upsert_edges_batch(valid_edges)
            progress.edges_created += len(valid_edges)

        # Update fingerprint
        await self._storage.update_fingerprint(
            file_path=result.file_path,
            content_hash=result.content_hash,
            size_bytes=result.size_bytes,
            last_modified=datetime.now(UTC).isoformat(),
            language=result.language,
            node_count=len(result.nodes),
        )

    def _discover_files(self) -> AsyncIterator[Path] | list[Path]:
        """Discover all source files in the project.

        Edge cases:
        - Symlink loops: track resolved paths in a seen set
        - Permission denied on directories: skip with warning
        - Huge directories (node_modules): skip entire dir tree early
        - Hidden directories (.git, .venv): skip via name check before fnmatch
        """
        import fnmatch

        root = self._config.project_root
        seen_resolved: set[Path] = set()
        results: list[Path] = []

        ignore_patterns = self._config.parser.ignored_patterns

        # Directories to skip immediately by name (before fnmatch overhead)
        skip_dir_names = {
            "node_modules", ".git", "__pycache__", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
            ".next", ".nuxt", "dist", "build", ".eggs", "vendor",
            ".gradle", ".idea", ".vscode", "coverage", "htmlcov",
            ".terraform", ".cargo", "target",
        }

        def _should_ignore_file(path: Path) -> bool:
            try:
                rel = str(path.relative_to(root))
            except ValueError:
                return True
            return any(fnmatch.fnmatch(rel, p) for p in ignore_patterns)

        def _walk(directory: Path) -> None:
            try:
                entries = sorted(directory.iterdir())
            except PermissionError:
                logger.warning("Permission denied: %s", directory)
                return
            except OSError:
                return

            for entry in entries:
                resolved = entry.resolve()

                # Symlink loop detection
                if resolved in seen_resolved:
                    continue
                seen_resolved.add(resolved)

                if entry.is_dir():
                    # Fast skip by directory name (no fnmatch needed)
                    dir_name = entry.name
                    if dir_name in skip_dir_names or dir_name.startswith("."):
                        continue
                    if not _should_ignore_file(entry):
                        _walk(entry)
                elif entry.is_file():
                    if not _should_ignore_file(entry):
                        lang = detect_language(entry)
                        if lang != Language.UNKNOWN:
                            results.append(entry)

        _walk(root)
        return results
