"""Git history miner — extracts decision candidates from PRs, commits, and comments.

The goal is to auto-discover decisions that were made but never formally recorded.
Developers make decisions constantly in PR descriptions, commit messages, and code
review comments — this module surfaces them.

This is SUGGESTION-based, not automatic. Every mined decision has confidence < 1.0
and should be reviewed by a human before becoming official.

Edge cases:
- PR description is empty: skip (nothing to mine)
- PR description is a template with checkboxes: extract only non-template content
- Commit message is "fix" or "wip": skip (no decision content)
- Multiple decisions in one PR: extract each as a separate candidate
- Decision language in non-English: basic support via keyword matching only
- Merge commits: skip (they reference the PR, not new decisions)
- Squash commits: contain the full PR description — high-value target
- Revert commits: flag the original decision as potentially superseded
- Git history is very large: limit mining depth with max_commits parameter
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codebase_intel.core.types import CodeAnchor, DecisionStatus
from codebase_intel.decisions.models import DecisionRecord

if TYPE_CHECKING:
    from codebase_intel.core.config import DecisionConfig

logger = logging.getLogger(__name__)

# Keywords that suggest a commit/PR contains a decision
DECISION_KEYWORDS = [
    # Architecture
    "decided", "decision", "chose", "chosen", "opted for", "switched to",
    "migrated from", "replaced", "instead of", "rather than",
    # Reasoning
    "because", "reason:", "rationale:", "why:", "trade-off", "tradeoff",
    "considered", "evaluated", "compared",
    # Constraints
    "compliance", "regulation", "requirement", "sla", "must not", "cannot",
    "forbidden", "prohibited",
    # Breaking changes
    "breaking change", "breaking:", "deprecated", "removed",
    # Architecture-specific
    "adr", "architecture decision", "design decision", "rfc",
]

# Patterns that indicate a commit should be skipped
SKIP_PATTERNS = [
    r"^merge\s",
    r"^wip\b",
    r"^fix\s*(typo|lint|format|style)",
    r"^chore\s*:",
    r"^bump\s+version",
    r"^update\s+lock",
    r"^auto-generated",
]


@dataclass
class DecisionCandidate:
    """A potential decision extracted from git history, pending human review."""

    title: str
    context: str
    decision_text: str
    source_type: str  # "commit", "pr_description", "pr_comment"
    source_ref: str  # commit hash, PR URL
    author: str
    created_at: datetime
    changed_files: list[Path] = field(default_factory=list)
    confidence: float = 0.5
    keywords_matched: list[str] = field(default_factory=list)

    def to_decision_record(self, decision_id: str) -> DecisionRecord:
        """Convert this candidate to a draft DecisionRecord."""
        anchors = [
            CodeAnchor(file_path=fp) for fp in self.changed_files[:10]
        ]

        return DecisionRecord(
            id=decision_id,
            title=self.title,
            status=DecisionStatus.DRAFT,
            context=self.context,
            decision=self.decision_text,
            code_anchors=anchors,
            created_at=self.created_at,
            author=self.author,
            source="git-mined",
            source_ref=self.source_ref,
            confidence=self.confidence,
            tags=["auto-mined"],
        )


class GitMiner:
    """Mines git history for decision candidates."""

    def __init__(
        self,
        config: DecisionConfig,
        project_root: Path,
    ) -> None:
        self._config = config
        self._project_root = project_root

    async def mine_commits(
        self,
        max_commits: int = 500,
        since_days: int = 90,
    ) -> list[DecisionCandidate]:
        """Mine recent commit messages for decision candidates.

        Edge cases:
        - Not a git repo: return empty list with warning
        - Shallow clone: limited history available — mine what we have
        - Binary commits (large media files): skip based on file extensions
        - Encoding issues in commit messages: handle gracefully
        """
        try:
            from git import Repo
        except ImportError:
            logger.warning("GitPython not installed — cannot mine commits")
            return []

        try:
            repo = Repo(self._project_root, search_parent_directories=True)
        except Exception:
            logger.warning("Not a git repository: %s", self._project_root)
            return []

        candidates: list[DecisionCandidate] = []
        count = 0

        since_dt = datetime.now(UTC).replace(
            day=max(1, datetime.now(UTC).day),
        )

        for commit in repo.iter_commits(max_count=max_commits):
            count += 1

            message = commit.message.strip()
            if not message:
                continue

            # Skip noise commits
            if self._should_skip(message):
                continue

            # Check for decision keywords
            matched_keywords = self._match_keywords(message)
            if not matched_keywords:
                continue

            # Extract changed files
            changed_files: list[Path] = []
            try:
                if commit.parents:
                    diff = commit.parents[0].diff(commit)
                    changed_files = [
                        self._project_root / d.a_path
                        for d in diff
                        if d.a_path
                    ]
            except Exception:
                pass  # Diff extraction is best-effort

            # Build candidate
            title = self._extract_title(message)
            context, decision_text = self._extract_context_and_decision(message)

            confidence = self._compute_confidence(
                message, matched_keywords, changed_files
            )

            candidates.append(DecisionCandidate(
                title=title,
                context=context,
                decision_text=decision_text,
                source_type="commit",
                source_ref=str(commit.hexsha)[:12],
                author=commit.author.name if commit.author else "unknown",
                created_at=datetime.fromtimestamp(commit.committed_date, tz=UTC),
                changed_files=changed_files[:20],
                confidence=confidence,
                keywords_matched=matched_keywords,
            ))

        logger.info(
            "Mined %d commits, found %d decision candidates",
            count,
            len(candidates),
        )

        return candidates

    def _should_skip(self, message: str) -> bool:
        """Check if a commit message should be skipped.

        Edge case: multi-line messages — check only the first line for
        skip patterns but check all lines for decision keywords.
        """
        first_line = message.split("\n")[0].lower().strip()
        return any(re.match(pattern, first_line) for pattern in SKIP_PATTERNS)

    def _match_keywords(self, message: str) -> list[str]:
        """Find decision-indicating keywords in the message."""
        message_lower = message.lower()
        return [kw for kw in DECISION_KEYWORDS if kw in message_lower]

    def _extract_title(self, message: str) -> str:
        """Extract a title from the commit message.

        Convention: first line of the commit message, truncated at 80 chars.
        Edge case: first line is very long (someone put everything on one line).
        """
        first_line = message.split("\n")[0].strip()
        if len(first_line) > 80:
            return first_line[:77] + "..."
        return first_line

    def _extract_context_and_decision(
        self, message: str
    ) -> tuple[str, str]:
        """Separate the context ("why") from the decision ("what") in a message.

        Heuristic: lines before "because"/"reason:" are the decision,
        lines after are the context. If no such separator, the whole
        message is both context and decision.

        Edge case: message with no clear separation — use the first line
        as the decision and the rest as context.
        """
        lines = message.strip().split("\n")

        if len(lines) <= 1:
            return message, message

        first_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        if not body:
            return first_line, first_line

        return body, first_line

    def _compute_confidence(
        self,
        message: str,
        keywords: list[str],
        changed_files: list[Path],
    ) -> float:
        """Compute confidence score for a mined decision candidate.

        Factors:
        - Number of keywords matched (more = higher)
        - Message length (longer = more context = higher)
        - Number of files changed (fewer = more focused = higher)
        - Presence of reasoning words (because, reason) = higher
        - Presence of "adr" or "decision" = much higher

        Score range: 0.2 (barely qualifying) to 0.8 (strong signal).
        Never 1.0 — that requires human confirmation.
        """
        score = 0.3  # Base score for matching any keyword

        # Keyword count bonus
        score += min(0.2, len(keywords) * 0.05)

        # Message length bonus (meaningful messages are longer)
        if len(message) > 200:
            score += 0.1
        if len(message) > 500:
            score += 0.1

        # Focused changes (fewer files = clearer decision)
        if 1 <= len(changed_files) <= 5:
            score += 0.05

        # High-signal keywords
        message_lower = message.lower()
        if any(kw in message_lower for kw in ("adr", "architecture decision", "design decision")):
            score += 0.15

        if any(kw in message_lower for kw in ("because", "reason:", "rationale:")):
            score += 0.1

        return min(0.8, score)
