"""Intent models — the schema for what users actually want.

The problem this solves:
  User: "I want live tracking"
  Agent: builds tracker.py (code exists)
  Agent: builds dashboard.py (code exists)
  Agent: doesn't wire them together (BROKEN)
  User discovers 2 hours later (WASTED)

With intent tracking:
  User: "I want live tracking"
  Intent captured with acceptance criteria:
    - "Every MCP call must record analytics"
    - "Dashboard must show real data from MCP usage"
  Before "done": system runs verification checks
  Result: "FAIL — analytics not wired into MCP server"
  Agent sees the gap and fixes it BEFORE claiming done.

Three levels of intent:
1. Session Intent — what the user wants from this coding session
2. Feature Intent — what a feature should do when complete
3. Project Intent — long-term goals and success criteria
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class IntentStatus(str, Enum):
    """Lifecycle of an intent."""

    ACTIVE = "active"  # Currently being worked on
    VERIFIED = "verified"  # All criteria met, confirmed working
    PARTIAL = "partial"  # Some criteria met, gaps remain
    FAILED = "failed"  # Verification ran, criteria not met
    ABANDONED = "abandoned"  # User decided not to pursue


class CriterionType(str, Enum):
    """How to verify an acceptance criterion."""

    FILE_EXISTS = "file_exists"  # A file must exist at this path
    FILE_CONTAINS = "file_contains"  # A file must contain this string/pattern
    FUNCTION_EXISTS = "function_exists"  # A function/class must exist in the graph
    WIRED = "wired"  # Module A must import/use module B (graph edge exists)
    CLI_WORKS = "cli_works"  # A CLI command must run without error
    MCP_TOOL_EXISTS = "mcp_tool_exists"  # An MCP tool must be registered
    GREP_MATCH = "grep_match"  # A pattern must match in the codebase
    GREP_NO_MATCH = "grep_no_match"  # A pattern must NOT match (removed something)
    TEST_PASSES = "test_passes"  # Specific test file/function must pass
    MANUAL = "manual"  # Human verification required (can't automate)
    CUSTOM = "custom"  # Custom shell command returns exit code 0


class AcceptanceCriterion(BaseModel):
    """A single verifiable acceptance criterion.

    Each criterion has:
    - A human description (what it means)
    - A machine-verifiable check (how to prove it)
    - A status (did it pass?)
    """

    model_config = ConfigDict(frozen=True)

    description: str = Field(description="Human-readable: what should be true")
    criterion_type: CriterionType
    check_value: str = Field(
        description=(
            "What to check. Depends on type:\n"
            "- file_exists: file path\n"
            "- file_contains: 'path::pattern' (file path :: regex)\n"
            "- function_exists: 'qualified.name'\n"
            "- wired: 'module_a -> module_b'\n"
            "- cli_works: 'command to run'\n"
            "- mcp_tool_exists: 'tool_name'\n"
            "- grep_match: 'pattern::path' (regex :: glob path)\n"
            "- grep_no_match: 'pattern::path'\n"
            "- test_passes: 'test_file::test_function'\n"
            "- manual: '' (human checks)\n"
            "- custom: 'shell command'"
        )
    )
    verified: bool = False
    verified_at: datetime | None = None
    failure_reason: str | None = None


class Intent(BaseModel):
    """A tracked intent with acceptance criteria.

    This is the core data structure. An intent captures:
    - WHAT the user wants (goal)
    - HOW to know it's done (acceptance criteria)
    - WHETHER it's actually done (verification status)
    - WHAT'S missing (gaps)
    """

    id: str = Field(description="Unique ID, e.g., 'INT-001'")
    title: str = Field(description="One-line summary of what's wanted")
    description: str = Field(
        default="",
        description="Detailed description of the goal",
    )
    status: IntentStatus = IntentStatus.ACTIVE
    priority: int = Field(
        default=1,
        ge=1,
        le=5,
        description="1=highest, 5=lowest",
    )

    # Acceptance criteria — the verifiable checklist
    criteria: list[AcceptanceCriterion] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    verified_at: datetime | None = None
    session_id: str | None = Field(
        default=None,
        description="MCP session that created this intent",
    )
    tags: list[str] = Field(default_factory=list)

    @field_validator("created_at", "updated_at", "verified_at")
    @classmethod
    def ensure_utc(cls, v: datetime | None) -> datetime | None:
        if v is None:
            return None
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)

    @property
    def criteria_met(self) -> int:
        return sum(1 for c in self.criteria if c.verified)

    @property
    def criteria_total(self) -> int:
        return len(self.criteria)

    @property
    def completion_pct(self) -> float:
        if not self.criteria:
            return 0.0
        return self.criteria_met / self.criteria_total * 100

    @property
    def gaps(self) -> list[AcceptanceCriterion]:
        """Criteria that haven't been met yet."""
        return [c for c in self.criteria if not c.verified]

    @property
    def is_complete(self) -> bool:
        return all(c.verified for c in self.criteria) if self.criteria else False

    def to_context_string(self) -> str:
        """Serialize for agent context — shows what's done and what's missing."""
        lines = [
            f"## Intent: {self.title} [{self.id}]",
            f"Status: {self.status.value} | Progress: {self.criteria_met}/{self.criteria_total} ({self.completion_pct:.0f}%)",
        ]

        if self.description:
            lines.append(f"\n{self.description}")

        if self.criteria:
            lines.append("\n**Acceptance Criteria:**")
            for c in self.criteria:
                icon = "[x]" if c.verified else "[ ]"
                lines.append(f"- {icon} {c.description}")
                if c.failure_reason:
                    lines.append(f"  FAILED: {c.failure_reason}")

        gaps = self.gaps
        if gaps:
            lines.append(f"\n**{len(gaps)} criteria still unmet — do NOT mark as done.**")

        return "\n".join(lines)
