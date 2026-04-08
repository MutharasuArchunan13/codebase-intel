"""Decision record models — the schema for capturing "why" in a machine-queryable format.

A decision record captures:
- What was decided
- Why (context, constraints, alternatives rejected)
- Where in the code it applies (code anchors)
- When it was made and when it should be reviewed
- What other decisions it relates to

Edge cases in schema design:
- Decision without code anchors: valid for org-level decisions ("we use Python 3.11+")
- Decision with anchors to deleted code: orphaned but possibly still relevant
- Decision that supersedes another: chain must be traversable
- Decision with conflicting constraints: e.g., "fast response" vs "complete validation"
- Decision with expiry date: temporary decisions ("use workaround until library v2")
- Cross-repo decisions: link format must support external repos
- Decision authored by an AI agent: must be flagged for human review
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codebase_intel.core.types import CodeAnchor, DecisionStatus


class AlternativeConsidered(BaseModel):
    """An option that was evaluated but not chosen.

    Recording rejected alternatives is critical — it prevents future
    developers (and AI agents) from proposing solutions that were
    already evaluated and dismissed.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    rejection_reason: str
    was_prototyped: bool = False  # Higher signal if it was actually tried


class Constraint(BaseModel):
    """A constraint that influenced the decision.

    Edge cases:
    - Compliance constraint: "GDPR requires data encryption at rest"
      → source=legal, is_hard=True
    - Performance constraint: "p99 < 200ms for this endpoint"
      → source=sla, is_hard=True
    - Preference: "team prefers explicit over magic"
      → source=team, is_hard=False
    - Temporary constraint: "budget freeze until Q3"
      → has expiry_date
    """

    model_config = ConfigDict(frozen=True)

    description: str
    source: str = Field(
        description="Where this constraint comes from: legal, sla, team, business, technical"
    )
    is_hard: bool = Field(
        default=True,
        description="Hard constraints can't be violated. Soft constraints are preferences.",
    )
    expiry_date: datetime | None = Field(
        default=None,
        description="If set, constraint is temporary and should be re-evaluated after this date",
    )

    @field_validator("expiry_date")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class DecisionRecord(BaseModel):
    """A single architectural or business decision.

    This is the core data structure of the Decision Journal.
    It's designed to be:
    - Human-writable (YAML-friendly)
    - Machine-queryable (structured fields)
    - Code-linked (anchored to specific locations)
    - Temporal (has a lifecycle with review dates)
    """

    model_config = ConfigDict(frozen=True)

    # --- Identity ---
    id: str = Field(description="Unique ID, e.g., 'DEC-042'")
    title: str = Field(description="Short summary, e.g., 'Use token bucket for rate limiting'")
    status: DecisionStatus = DecisionStatus.ACTIVE

    # --- Context ---
    context: str = Field(
        description="The situation that prompted this decision — what problem were we solving?"
    )
    decision: str = Field(
        description="What was decided — the actual choice made"
    )
    consequences: list[str] = Field(
        default_factory=list,
        description="Known consequences (positive and negative) of this decision",
    )
    alternatives: list[AlternativeConsidered] = Field(
        default_factory=list,
        description="Options that were evaluated and rejected",
    )
    constraints: list[Constraint] = Field(
        default_factory=list,
        description="Constraints that shaped this decision",
    )

    # --- Code linking ---
    code_anchors: list[CodeAnchor] = Field(
        default_factory=list,
        description="Specific code locations this decision applies to",
    )

    # --- Relationships ---
    supersedes: str | None = Field(
        default=None,
        description="ID of the decision this one replaces",
    )
    related_decisions: list[str] = Field(
        default_factory=list,
        description="IDs of related (but not superseded) decisions",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Categorization tags: 'architecture', 'security', 'performance', etc.",
    )

    # --- Temporal ---
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    review_by: datetime | None = Field(
        default=None,
        description="Date by which this decision should be reviewed for continued relevance",
    )
    last_validated: datetime | None = Field(
        default=None,
        description="Last time someone confirmed this decision still applies",
    )

    # --- Provenance ---
    author: str = Field(
        default="unknown",
        description="Who made this decision (person or team name)",
    )
    source: str = Field(
        default="manual",
        description="How this record was created: manual, git-mined, ai-suggested",
    )
    source_ref: str | None = Field(
        default=None,
        description="Reference to source: PR URL, commit hash, meeting notes link",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident we are in this record: 1.0=verified, <1.0=auto-extracted",
    )

    # --- Validation ---

    @field_validator("created_at", "review_by", "last_validated")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @model_validator(mode="after")
    def validate_supersedes_chain(self) -> DecisionRecord:
        """Edge case: a decision cannot supersede itself."""
        if self.supersedes == self.id:
            msg = f"Decision {self.id} cannot supersede itself"
            raise ValueError(msg)
        return self

    @property
    def is_stale(self) -> bool:
        """Check if this decision is past its review date.

        Edge case: review_by not set → not stale (no review requested).
        """
        if self.review_by is None:
            return False
        return datetime.now(UTC) > self.review_by

    @property
    def is_expired(self) -> bool:
        """Check if any hard constraints have expired.

        Edge case: expired constraint doesn't automatically invalidate the decision
        (the decision may still be valid for other reasons). It flags for review.
        """
        return any(
            c.expiry_date is not None and datetime.now(UTC) > c.expiry_date
            for c in self.constraints
            if c.is_hard
        )

    @property
    def has_orphaned_anchors(self) -> bool:
        """Check if any code anchors reference files that don't exist."""
        return any(
            not anchor.file_path.exists()
            for anchor in self.code_anchors
        )

    def relevance_score(self, file_paths: set[Path]) -> float:
        """Score how relevant this decision is to a set of files.

        Scoring:
        - Direct anchor match: 1.0
        - Same directory as an anchor: 0.5
        - Same top-level package: 0.2
        - No match: 0.0

        Edge case: decision with no anchors (org-level) gets a baseline
        score of 0.1 — always slightly relevant.
        """
        if not self.code_anchors:
            return 0.1  # Org-level decision, always slightly relevant

        max_score = 0.0
        anchor_paths = {a.file_path.resolve() for a in self.code_anchors}
        resolved_files = {p.resolve() for p in file_paths}

        for anchor_path in anchor_paths:
            # Direct match
            if anchor_path in resolved_files:
                return 1.0

            # Same directory
            for fp in resolved_files:
                if anchor_path.parent == fp.parent:
                    max_score = max(max_score, 0.5)
                elif (
                    len(anchor_path.parts) >= 3
                    and len(fp.parts) >= 3
                    and anchor_path.parts[:3] == fp.parts[:3]
                ):
                    max_score = max(max_score, 0.2)

        return max_score

    def to_context_string(self, verbose: bool = False) -> str:
        """Serialize for inclusion in agent context.

        Two modes:
        - Compact: title + decision + constraints (fits in tight budgets)
        - Verbose: full record including alternatives and consequences
        """
        lines = [
            f"## Decision: {self.title} [{self.id}]",
            f"Status: {self.status.value}",
            f"",
            f"**Context:** {self.context}",
            f"**Decision:** {self.decision}",
        ]

        if self.constraints:
            lines.append("")
            lines.append("**Constraints:**")
            for c in self.constraints:
                hard = "MUST" if c.is_hard else "SHOULD"
                lines.append(f"- [{hard}] {c.description} (source: {c.source})")

        if verbose:
            if self.alternatives:
                lines.append("")
                lines.append("**Alternatives considered:**")
                for alt in self.alternatives:
                    lines.append(f"- {alt.name}: {alt.rejection_reason}")

            if self.consequences:
                lines.append("")
                lines.append("**Consequences:**")
                for c in self.consequences:
                    lines.append(f"- {c}")

        if self.is_stale:
            lines.append("")
            lines.append(f"**WARNING: This decision is past its review date ({self.review_by})**")

        return "\n".join(lines)
