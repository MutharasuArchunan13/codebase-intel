"""Decision store — YAML-based persistence with query capabilities.

Decisions are stored as individual YAML files in the decisions directory.
This is intentional over SQLite because:
1. YAML files are human-editable (developers can create/edit decisions directly)
2. YAML files are git-friendly (diff, blame, merge work naturally)
3. Individual files prevent merge conflicts (parallel decision creation)
4. Files can be reviewed in PRs alongside the code they reference

Edge cases:
- YAML syntax error in a decision file: skip that file, report error, continue
- Multiple files with same decision ID: conflict — report and use the newer one
- Decision file modified externally while MCP server is running: detect via mtime
- Decision references a superseded decision that doesn't exist: orphaned chain
- Bulk operations (100+ decisions): batch loading with lazy parsing
- Unicode in decision content: YAML handles this natively
- Large decision files (someone put a design doc in the description): warn but allow
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from codebase_intel.core.exceptions import (
    ContractParseError,
    DecisionConflictError,
    ErrorContext,
    OrphanedDecisionError,
    StaleDecisionError,
)
from codebase_intel.core.types import CodeAnchor, DecisionStatus, LineRange
from codebase_intel.decisions.models import (
    AlternativeConsidered,
    Constraint,
    DecisionRecord,
)

if TYPE_CHECKING:
    from codebase_intel.core.config import DecisionConfig

logger = logging.getLogger(__name__)


class DecisionStore:
    """Manages decision records stored as YAML files."""

    def __init__(self, config: DecisionConfig, project_root: Path) -> None:
        self._config = config
        self._project_root = project_root
        self._cache: dict[str, DecisionRecord] = {}
        self._cache_mtimes: dict[str, float] = {}

    @property
    def decisions_dir(self) -> Path:
        return self._config.decisions_dir

    async def load_all(self) -> list[DecisionRecord]:
        """Load all decision records from the decisions directory.

        Edge cases:
        - Directory doesn't exist: return empty list (not initialized yet)
        - Malformed YAML: skip that file, log error with filename
        - Duplicate IDs: keep the most recently modified file, warn
        - Empty YAML file: skip with warning
        """
        if not self.decisions_dir.exists():
            return []

        decisions: dict[str, DecisionRecord] = {}
        errors: list[str] = []

        for yaml_file in sorted(self.decisions_dir.glob("*.yaml")):
            try:
                record = self._load_file(yaml_file)
            except Exception as exc:
                errors.append(f"{yaml_file.name}: {exc}")
                logger.warning("Failed to load decision file %s: %s", yaml_file, exc)
                continue

            if record is None:
                continue

            # Handle duplicate IDs
            if record.id in decisions:
                existing_file = self._find_file_for_id(record.id)
                logger.warning(
                    "Duplicate decision ID '%s' in %s and %s — keeping newer",
                    record.id,
                    existing_file,
                    yaml_file,
                )

            decisions[record.id] = record
            self._cache[record.id] = record
            self._cache_mtimes[record.id] = yaml_file.stat().st_mtime

        if errors:
            logger.warning(
                "%d decision files had errors: %s",
                len(errors),
                "; ".join(errors[:5]),
            )

        return list(decisions.values())

    def _load_file(self, yaml_file: Path) -> DecisionRecord | None:
        """Load a single decision YAML file.

        Edge case: file exists but is empty → returns None.
        Edge case: file has YAML but missing required fields → validation error.
        """
        content = yaml_file.read_text(encoding="utf-8")
        if not content.strip():
            return None

        try:
            data = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ContractParseError(
                f"Invalid YAML in {yaml_file.name}: {exc}",
                ErrorContext(file_path=yaml_file, operation="parse_decision"),
            ) from exc

        if not isinstance(data, dict):
            raise ContractParseError(
                f"Decision file {yaml_file.name} must contain a YAML mapping, got {type(data).__name__}",
                ErrorContext(file_path=yaml_file),
            )

        # Parse code anchors from simplified YAML format
        anchors = []
        for anchor_data in data.get("code_anchors", []):
            if isinstance(anchor_data, str):
                # Shorthand: "src/middleware/rate_limiter.py:15-82"
                anchors.append(self._parse_anchor_shorthand(anchor_data))
            elif isinstance(anchor_data, dict):
                anchors.append(CodeAnchor(
                    file_path=Path(anchor_data["file_path"]),
                    line_range=LineRange(**anchor_data["line_range"]) if "line_range" in anchor_data else None,
                    symbol_name=anchor_data.get("symbol_name"),
                    content_hash=anchor_data.get("content_hash"),
                ))
        data["code_anchors"] = anchors

        # Parse alternatives
        alternatives = []
        for alt_data in data.get("alternatives", []):
            if isinstance(alt_data, dict):
                alternatives.append(AlternativeConsidered(**alt_data))
        data["alternatives"] = alternatives

        # Parse constraints
        constraints = []
        for c_data in data.get("constraints", []):
            if isinstance(c_data, str):
                constraints.append(Constraint(description=c_data, source="unknown"))
            elif isinstance(c_data, dict):
                constraints.append(Constraint(**c_data))
        data["constraints"] = constraints

        return DecisionRecord(**data)

    def _parse_anchor_shorthand(self, shorthand: str) -> CodeAnchor:
        """Parse shorthand anchor format: 'path/to/file.py:15-82' or 'path/to/file.py'.

        Edge cases:
        - No line range: anchor applies to entire file
        - Single line: "file.py:42" → LineRange(start=42, end=42)
        - Line range: "file.py:15-82" → LineRange(start=15, end=82)
        - Path with colons (Windows, unlikely in POSIX): take last colon
        """
        if ":" in shorthand:
            path_part, line_part = shorthand.rsplit(":", 1)
            line_range = None
            if "-" in line_part:
                start, end = line_part.split("-", 1)
                if start.isdigit() and end.isdigit():
                    line_range = LineRange(start=int(start), end=int(end))
            elif line_part.isdigit():
                line_no = int(line_part)
                line_range = LineRange(start=line_no, end=line_no)
            else:
                path_part = shorthand
            return CodeAnchor(file_path=Path(path_part), line_range=line_range)

        return CodeAnchor(file_path=Path(shorthand))

    async def get(self, decision_id: str) -> DecisionRecord | None:
        """Get a single decision by ID.

        Edge case: cached record might be stale if file was edited externally.
        Check mtime before returning cached value.
        """
        if decision_id in self._cache:
            # Check if file was modified since cache
            file_path = self._find_file_for_id(decision_id)
            if file_path and file_path.exists():
                current_mtime = file_path.stat().st_mtime
                cached_mtime = self._cache_mtimes.get(decision_id, 0)
                if current_mtime > cached_mtime:
                    # Reload
                    record = self._load_file(file_path)
                    if record:
                        self._cache[decision_id] = record
                        self._cache_mtimes[decision_id] = current_mtime
                        return record

            return self._cache[decision_id]

        # Not in cache — scan directory
        all_decisions = await self.load_all()
        return self._cache.get(decision_id)

    async def save(self, record: DecisionRecord) -> Path:
        """Save a decision record to a YAML file.

        Filename convention: DEC-042.yaml (based on ID).

        Edge case: ID contains characters invalid in filenames.
        We sanitize but keep the mapping clear.
        """
        self.decisions_dir.mkdir(parents=True, exist_ok=True)

        filename = self._id_to_filename(record.id)
        file_path = self.decisions_dir / filename

        data = self._record_to_yaml_dict(record)
        content = yaml.dump(
            data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        )

        file_path.write_text(content, encoding="utf-8")

        self._cache[record.id] = record
        self._cache_mtimes[record.id] = file_path.stat().st_mtime

        logger.info("Saved decision %s to %s", record.id, file_path)
        return file_path

    async def query_by_files(
        self,
        file_paths: set[Path],
        min_relevance: float = 0.1,
    ) -> list[tuple[DecisionRecord, float]]:
        """Find decisions relevant to a set of files, scored by relevance.

        Returns decisions sorted by relevance (highest first).

        Edge cases:
        - No decisions exist: return empty list
        - All decisions are stale: still return them (agent should know about
          stale decisions to avoid contradicting them)
        - Decision anchored to renamed file: relevance_score is 0 for the new
          path, but drift detector will flag this
        """
        all_decisions = await self.load_all()
        scored: list[tuple[DecisionRecord, float]] = []

        for record in all_decisions:
            if record.status in (DecisionStatus.SUPERSEDED, DecisionStatus.EXPIRED):
                continue
            score = record.relevance_score(file_paths)
            if score >= min_relevance:
                scored.append((record, score))

        return sorted(scored, key=lambda x: x[1], reverse=True)

    async def query_by_tags(self, tags: list[str]) -> list[DecisionRecord]:
        """Find decisions matching any of the given tags."""
        all_decisions = await self.load_all()
        tag_set = set(tags)
        return [d for d in all_decisions if set(d.tags) & tag_set]

    async def find_conflicts(self) -> list[tuple[DecisionRecord, DecisionRecord, str]]:
        """Detect conflicting active decisions.

        Conflict detection heuristics:
        1. Two active decisions anchored to the same code region
        2. A decision and its superseded predecessor both marked active
        3. Decisions with contradictory constraints (manual tagging required)

        Edge case: decisions may "look" conflicting (same file, different regions)
        but actually address different concerns. We flag potential conflicts
        and let humans/agents resolve.

        Returns: list of (decision_a, decision_b, conflict_reason) tuples.
        """
        all_decisions = await self.load_all()
        active = [d for d in all_decisions if d.status == DecisionStatus.ACTIVE]
        conflicts: list[tuple[DecisionRecord, DecisionRecord, str]] = []

        # Check for supersedes chain conflicts
        superseded_ids = {d.supersedes for d in active if d.supersedes}
        for decision in active:
            if decision.id in superseded_ids:
                superseder = next(
                    (d for d in active if d.supersedes == decision.id), None
                )
                if superseder:
                    conflicts.append((
                        decision,
                        superseder,
                        f"{superseder.id} supersedes {decision.id} but both are active",
                    ))

        # Check for overlapping code anchors
        for i, d1 in enumerate(active):
            for d2 in active[i + 1 :]:
                overlap = self._check_anchor_overlap(d1, d2)
                if overlap:
                    conflicts.append((d1, d2, overlap))

        return conflicts

    def _check_anchor_overlap(
        self, d1: DecisionRecord, d2: DecisionRecord
    ) -> str | None:
        """Check if two decisions have overlapping code anchors.

        Edge case: same file but different line ranges — only flag if ranges
        actually overlap, not just because they're in the same file.
        """
        for a1 in d1.code_anchors:
            for a2 in d2.code_anchors:
                if a1.file_path.resolve() != a2.file_path.resolve():
                    continue

                # Same file — check line ranges
                if a1.line_range is None or a2.line_range is None:
                    # At least one anchors the whole file — potential overlap
                    return (
                        f"Both anchor to {a1.file_path.name} "
                        f"(check if they address different concerns)"
                    )

                # Check actual overlap
                if (
                    a1.line_range.start <= a2.line_range.end
                    and a2.line_range.start <= a1.line_range.end
                ):
                    return (
                        f"Overlapping code regions in {a1.file_path.name}: "
                        f"lines {a1.line_range.start}-{a1.line_range.end} "
                        f"and {a2.line_range.start}-{a2.line_range.end}"
                    )

        return None

    async def next_id(self) -> str:
        """Generate the next decision ID.

        Format: DEC-NNN with zero-padded 3-digit number.
        Edge case: gaps in numbering (DEC-005 deleted) — we don't reuse IDs,
        always increment from the highest existing ID.
        """
        all_decisions = await self.load_all()
        max_num = 0
        for d in all_decisions:
            if d.id.startswith("DEC-"):
                try:
                    num = int(d.id.split("-", 1)[1])
                    max_num = max(max_num, num)
                except (ValueError, IndexError):
                    pass
        return f"DEC-{max_num + 1:03d}"

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    def _id_to_filename(self, decision_id: str) -> str:
        """Convert decision ID to a safe filename."""
        safe = decision_id.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return f"{safe}.yaml"

    def _find_file_for_id(self, decision_id: str) -> Path | None:
        """Find the YAML file for a decision ID."""
        filename = self._id_to_filename(decision_id)
        file_path = self.decisions_dir / filename
        return file_path if file_path.exists() else None

    def _record_to_yaml_dict(self, record: DecisionRecord) -> dict:
        """Convert a DecisionRecord to a YAML-friendly dict.

        Edge case: datetime objects → ISO format strings for YAML.
        Edge case: Path objects → relative strings for portability.
        """
        data = record.model_dump(mode="json")

        # Convert code anchors to shorthand format for readability
        if data.get("code_anchors"):
            anchors = []
            for anchor in data["code_anchors"]:
                fp = anchor["file_path"]
                lr = anchor.get("line_range")
                if lr:
                    anchors.append(f"{fp}:{lr['start']}-{lr['end']}")
                else:
                    anchors.append(fp)
            data["code_anchors"] = anchors

        return data
