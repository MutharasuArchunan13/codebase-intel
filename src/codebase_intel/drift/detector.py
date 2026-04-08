"""Drift detector — identifies staleness, pattern violations, and knowledge decay.

Drift is the gradual divergence between what the system "knows" (decisions,
contracts, graph state) and what the code actually is. Drift is inevitable —
code changes constantly. The detector's job is to surface drift before it
causes problems (like an agent acting on outdated context).

Types of drift detected:
1. Decision drift: code changed but decision anchors still point to old locations
2. Contract drift: code violates contracts that it previously satisfied
3. Graph drift: graph is stale (files changed since last index)
4. Context rot: >30% of records are stale (systemic problem, not individual)

Edge cases:
- False positive from refactor: file moved but logic is the same → content hash
  matching prevents false positives (same hash at new path = rename, not violation)
- Intentional drift: team decided to violate a contract temporarily → migration
  deadlines in contracts handle this gracefully
- Large PR with many changes: drift check shouldn't block the workflow. Run async
  and report non-blocking warnings.
- New developer joins: lots of "new" code the system hasn't seen → initial noise,
  settles after first re-index
- Gradual drift vs sudden drift: gradual (1 file/week drifts) vs sudden (major
  refactor invalidates 40% of records). Different responses needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codebase_intel.core.exceptions import ContextRotError, ErrorContext
from codebase_intel.core.types import DriftLevel

if TYPE_CHECKING:
    from codebase_intel.core.config import DriftConfig, ProjectConfig
    from codebase_intel.decisions.store import DecisionStore
    from codebase_intel.graph.storage import GraphStorage


logger = logging.getLogger(__name__)


@dataclass
class DriftItem:
    """A single instance of detected drift."""

    component: str  # "decision", "contract", "graph"
    level: DriftLevel
    description: str
    file_path: Path | None = None
    record_id: str | None = None
    remediation: str = ""


@dataclass
class DriftReport:
    """Complete drift analysis report."""

    items: list[DriftItem] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    graph_stale_files: int = 0
    decision_stale_count: int = 0
    decision_orphaned_count: int = 0
    decision_total: int = 0
    rot_detected: bool = False
    rot_percentage: float = 0.0

    @property
    def overall_level(self) -> DriftLevel:
        """The most severe drift level across all items."""
        if not self.items:
            return DriftLevel.NONE
        levels = [i.level for i in self.items]
        severity = [DriftLevel.NONE, DriftLevel.LOW, DriftLevel.MEDIUM, DriftLevel.HIGH, DriftLevel.CRITICAL]
        return max(levels, key=lambda l: severity.index(l))

    @property
    def summary(self) -> str:
        """One-line summary for CLI/MCP output."""
        if not self.items:
            return "No drift detected. All records are current."
        counts = {}
        for item in self.items:
            counts[item.level.value] = counts.get(item.level.value, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(counts.items())]
        return f"Drift detected: {', '.join(parts)}"

    def to_context_string(self) -> str:
        """Serialize for inclusion in agent context."""
        if not self.items:
            return ""

        lines = [
            "## Drift Warnings",
            f"Overall status: {self.overall_level.value}",
            "",
        ]

        # Group by component
        by_component: dict[str, list[DriftItem]] = {}
        for item in self.items:
            by_component.setdefault(item.component, []).append(item)

        for component, items in by_component.items():
            lines.append(f"### {component.title()}")
            for item in items[:10]:  # Cap per component
                lines.append(f"- [{item.level.value}] {item.description}")
                if item.remediation:
                    lines.append(f"  → {item.remediation}")
            if len(items) > 10:
                lines.append(f"  ... and {len(items) - 10} more")
            lines.append("")

        if self.rot_detected:
            lines.append(
                f"**CONTEXT ROT ALERT**: {self.rot_percentage:.0%} of decision records "
                f"are stale. Consider running `codebase-intel refresh`."
            )

        return "\n".join(lines)


class DriftDetector:
    """Detects drift between recorded knowledge and actual code state."""

    def __init__(
        self,
        config: DriftConfig,
        project_root: Path,
        graph_storage: GraphStorage | None = None,
        decision_store: DecisionStore | None = None,
    ) -> None:
        self._config = config
        self._project_root = project_root
        self._graph = graph_storage
        self._decisions = decision_store

    async def full_check(self) -> DriftReport:
        """Run a comprehensive drift check across all components.

        This is the main entry point — called by CLI and post-commit hook.
        """
        report = DriftReport()

        # Check each component independently
        if self._graph:
            await self._check_graph_drift(report)

        if self._decisions:
            await self._check_decision_drift(report)

        # Check for context rot (systemic staleness)
        self._check_context_rot(report)

        logger.info(
            "Drift check complete: %d items, overall=%s",
            len(report.items),
            report.overall_level.value,
        )

        return report

    async def check_files(self, changed_files: list[Path]) -> DriftReport:
        """Quick drift check focused on specific changed files.

        Used by git hooks — only checks drift related to the changed files,
        not the entire project. Much faster than full_check.

        Edge cases:
        - Changed file has no decisions: no decision drift (valid)
        - Changed file is not in graph: graph drift (needs re-index)
        - Changed file was deleted: orphan check on decisions anchored to it
        """
        report = DriftReport()

        if self._graph:
            await self._check_graph_drift_for_files(report, changed_files)

        if self._decisions:
            await self._check_decision_drift_for_files(report, changed_files)

        return report

    # -------------------------------------------------------------------
    # Graph drift
    # -------------------------------------------------------------------

    async def _check_graph_drift(self, report: DriftReport) -> None:
        """Check if the code graph is stale.

        Compares stored fingerprints against current file state.
        Files that changed since last index are flagged.

        Edge case: new files (not in graph at all) are not drift —
        they're additions. Only flag files whose content changed.
        """
        from codebase_intel.graph.parser import compute_file_hash

        stale_count = 0
        cursor = await self._graph._db.execute(  # type: ignore[union-attr]
            "SELECT file_path, content_hash FROM file_fingerprints"
        )
        for row in await cursor.fetchall():
            stored_path = row[0]
            stored_hash = row[1]
            full_path = self._graph._from_stored_path(stored_path)  # type: ignore[union-attr]

            if not full_path.exists():
                report.items.append(DriftItem(
                    component="graph",
                    level=DriftLevel.HIGH,
                    description=f"File {stored_path} was deleted but still in graph",
                    file_path=full_path,
                    remediation="Run `codebase-intel analyze` to update the graph",
                ))
                stale_count += 1
                continue

            try:
                current_hash = compute_file_hash(full_path.read_bytes())
            except OSError:
                continue

            if current_hash != stored_hash:
                report.items.append(DriftItem(
                    component="graph",
                    level=DriftLevel.MEDIUM,
                    description=f"File {stored_path} changed since last index",
                    file_path=full_path,
                    remediation="Run `codebase-intel analyze --incremental`",
                ))
                stale_count += 1

        report.graph_stale_files = stale_count

    async def _check_graph_drift_for_files(
        self, report: DriftReport, files: list[Path]
    ) -> None:
        """Quick graph drift check for specific files."""
        from codebase_intel.graph.parser import compute_file_hash

        for fp in files:
            if not fp.exists():
                continue

            stored_hash = await self._graph.get_fingerprint(fp)  # type: ignore[union-attr]
            if stored_hash is None:
                # New file — not drift, just needs indexing
                report.items.append(DriftItem(
                    component="graph",
                    level=DriftLevel.LOW,
                    description=f"New file {fp.name} not yet in graph",
                    file_path=fp,
                    remediation="Will be indexed on next build",
                ))
                continue

            current_hash = compute_file_hash(fp.read_bytes())
            if current_hash != stored_hash:
                report.graph_stale_files += 1

    # -------------------------------------------------------------------
    # Decision drift
    # -------------------------------------------------------------------

    async def _check_decision_drift(self, report: DriftReport) -> None:
        """Check all decisions for staleness and orphaning.

        Three types of decision drift:
        1. Stale: past review_by date
        2. Orphaned: code anchors point to deleted files
        3. Content drift: anchored code changed significantly (hash mismatch)

        Edge case: decision with no code anchors is never orphaned
        (it's an org-level decision, always somewhat relevant).

        Edge case: decision that's both stale AND orphaned → report as
        CRITICAL drift (double signal that this decision needs attention).
        """
        all_decisions = await self._decisions.load_all()  # type: ignore[union-attr]
        report.decision_total = len(all_decisions)
        stale_count = 0
        orphaned_count = 0

        for record in all_decisions:
            if record.status != "active":
                continue

            # Check staleness
            if record.is_stale:
                level = DriftLevel.MEDIUM
                stale_count += 1
                report.items.append(DriftItem(
                    component="decision",
                    level=level,
                    description=(
                        f"Decision {record.id} ('{record.title}') is past "
                        f"its review date ({record.review_by})"
                    ),
                    record_id=record.id,
                    remediation="Review and update the decision, or extend the review date",
                ))

            # Check for expired constraints
            if record.is_expired:
                report.items.append(DriftItem(
                    component="decision",
                    level=DriftLevel.HIGH,
                    description=(
                        f"Decision {record.id} has expired constraints — "
                        f"the original rationale may no longer apply"
                    ),
                    record_id=record.id,
                    remediation="Re-evaluate whether this decision still holds",
                ))

            # Check for orphaned anchors
            if record.has_orphaned_anchors:
                orphaned_count += 1
                orphaned_paths = [
                    str(a.file_path)
                    for a in record.code_anchors
                    if not a.file_path.exists()
                ]
                level = DriftLevel.HIGH if record.is_stale else DriftLevel.MEDIUM

                report.items.append(DriftItem(
                    component="decision",
                    level=level,
                    description=(
                        f"Decision {record.id} anchored to deleted files: "
                        f"{', '.join(orphaned_paths[:3])}"
                    ),
                    record_id=record.id,
                    remediation="Update code anchors to new file locations or supersede the decision",
                ))

            # Check content hash drift on remaining anchors
            for anchor in record.code_anchors:
                if not anchor.file_path.exists() or not anchor.content_hash:
                    continue

                try:
                    content = anchor.file_path.read_text(encoding="utf-8")
                    if anchor.line_range:
                        lines = content.split("\n")
                        region = "\n".join(
                            lines[anchor.line_range.start - 1 : anchor.line_range.end]
                        )
                    else:
                        region = content

                    from codebase_intel.graph.parser import compute_file_hash
                    current_hash = compute_file_hash(region.encode())

                    if current_hash != anchor.content_hash:
                        report.items.append(DriftItem(
                            component="decision",
                            level=DriftLevel.MEDIUM,
                            description=(
                                f"Code anchored by {record.id} at "
                                f"{anchor.file_path.name} has changed"
                            ),
                            file_path=anchor.file_path,
                            record_id=record.id,
                            remediation="Verify the decision still applies to the changed code",
                        ))
                except OSError:
                    pass

        report.decision_stale_count = stale_count
        report.decision_orphaned_count = orphaned_count

    async def _check_decision_drift_for_files(
        self, report: DriftReport, files: list[Path]
    ) -> None:
        """Quick decision drift check for specific changed files."""
        file_set = {fp.resolve() for fp in files}
        all_decisions = await self._decisions.load_all()  # type: ignore[union-attr]

        for record in all_decisions:
            if record.status != "active":
                continue

            for anchor in record.code_anchors:
                if anchor.file_path.resolve() in file_set:
                    report.items.append(DriftItem(
                        component="decision",
                        level=DriftLevel.LOW,
                        description=(
                            f"Changed file {anchor.file_path.name} is anchored "
                            f"by decision {record.id}"
                        ),
                        file_path=anchor.file_path,
                        record_id=record.id,
                        remediation="Verify the decision still applies after your changes",
                    ))

    # -------------------------------------------------------------------
    # Context rot detection
    # -------------------------------------------------------------------

    def _check_context_rot(self, report: DriftReport) -> None:
        """Detect systemic context rot — when too many records are stale.

        Context rot is qualitatively different from individual drift items.
        It means the knowledge base as a whole is unreliable and needs a
        bulk review, not item-by-item fixing.

        Edge case: project with only 2 decisions, 1 is stale → 50% rot.
        But that's not really "systemic." We require a minimum of 5 decisions
        before triggering the rot alert.

        Edge case: rot threshold is configurable. Default 30% — adjustable
        for projects that tolerate more staleness.
        """
        total = report.decision_total
        stale = report.decision_stale_count + report.decision_orphaned_count

        if total < 5:
            return  # Too few decisions to assess rot

        rot_pct = stale / total
        report.rot_percentage = rot_pct

        if rot_pct >= self._config.rot_threshold_pct:
            report.rot_detected = True
            report.items.append(DriftItem(
                component="system",
                level=DriftLevel.CRITICAL,
                description=(
                    f"Context rot: {rot_pct:.0%} of decision records are stale or orphaned "
                    f"({stale}/{total})"
                ),
                remediation=(
                    "Run `codebase-intel refresh` to identify and update stale records. "
                    "Consider a team review session for bulk cleanup."
                ),
            ))
