"""Context quality feedback loop — learns from agent output acceptance/rejection.

THIS IS THE MOAT. No other tool does this.

The problem: every context tool sends context and hopes for the best.
Nobody tracks whether the context actually helped. Did the agent's output
get accepted? Rejected? Modified heavily?

This module closes the loop:
1. When context is assembled, record a "session"
2. When the agent's output is reviewed (accepted/rejected/modified), record feedback
3. Over time, learn which context patterns lead to acceptance:
   - "Decisions about auth always improve agent output" → boost priority
   - "The AI guardrails contract catches 80% of rejections" → prove value
   - "Files from the graph's depth-2 traversal are rarely useful" → trim context
4. Surface insights: "Your AI output acceptance rate improved from 62% to 84%
   after adding quality contracts"

This is tracked per-project and powers the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackType(str, Enum):
    """How the agent's output was received."""

    ACCEPTED = "accepted"  # Output used as-is or with minor edits
    MODIFIED = "modified"  # Output used but significantly changed
    REJECTED = "rejected"  # Output discarded entirely
    PARTIAL = "partial"  # Some parts used, some discarded


class RejectionReason(str, Enum):
    """Why the output was rejected — maps to improvement actions."""

    WRONG_PATTERN = "wrong_pattern"  # Used a pattern the project doesn't follow
    MISSING_CONTEXT = "missing_context"  # Didn't know about a dependency/constraint
    VIOLATED_DECISION = "violated_decision"  # Contradicted an existing decision
    HALLUCINATED = "hallucinated"  # Referenced non-existent API/module
    OVER_ENGINEERED = "over_engineered"  # Too complex for the task
    SECURITY_ISSUE = "security_issue"  # Introduced a vulnerability
    WRONG_APPROACH = "wrong_approach"  # Fundamentally wrong solution
    OTHER = "other"


FEEDBACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    task_description TEXT NOT NULL,
    files_involved TEXT NOT NULL DEFAULT '[]',
    context_event_id INTEGER,
    tokens_used INTEGER NOT NULL DEFAULT 0,
    decisions_provided INTEGER NOT NULL DEFAULT 0,
    contracts_provided INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES context_sessions(session_id),
    timestamp TEXT NOT NULL,
    feedback_type TEXT NOT NULL,
    rejection_reason TEXT,
    details TEXT,
    files_affected TEXT DEFAULT '[]',
    improvement_suggestion TEXT
);

CREATE TABLE IF NOT EXISTS learning_insights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    insight_type TEXT NOT NULL,
    description TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    action_taken TEXT,
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS acceptance_trend (
    date TEXT PRIMARY KEY,
    total_sessions INTEGER NOT NULL DEFAULT 0,
    accepted INTEGER NOT NULL DEFAULT 0,
    modified INTEGER NOT NULL DEFAULT 0,
    rejected INTEGER NOT NULL DEFAULT 0,
    partial INTEGER NOT NULL DEFAULT 0,
    acceptance_rate REAL NOT NULL DEFAULT 0
);
"""


class FeedbackTracker:
    """Tracks context quality and learns from agent output reception."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(FEEDBACK_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def start_session(
        self,
        session_id: str,
        task_description: str,
        files_involved: list[str],
        context_event_id: int | None = None,
        tokens_used: int = 0,
        decisions_provided: int = 0,
        contracts_provided: int = 0,
    ) -> None:
        """Record that context was provided to an agent."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO context_sessions (
                session_id, timestamp, task_description, files_involved,
                context_event_id, tokens_used, decisions_provided, contracts_provided
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now(UTC).isoformat(),
                task_description[:500],
                json.dumps(files_involved[:20]),
                context_event_id,
                tokens_used,
                decisions_provided,
                contracts_provided,
            ),
        )
        conn.commit()

    def record_feedback(
        self,
        session_id: str,
        feedback_type: FeedbackType,
        rejection_reason: RejectionReason | None = None,
        details: str | None = None,
        files_affected: list[str] | None = None,
        improvement_suggestion: str | None = None,
    ) -> None:
        """Record how the agent's output was received.

        This is called by the MCP tool `record_feedback` — the agent or
        user reports whether the generated code was useful.
        """
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO feedback_events (
                session_id, timestamp, feedback_type, rejection_reason,
                details, files_affected, improvement_suggestion
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                datetime.now(UTC).isoformat(),
                feedback_type.value,
                rejection_reason.value if rejection_reason else None,
                details,
                json.dumps(files_affected or []),
                improvement_suggestion,
            ),
        )
        conn.commit()
        self._update_acceptance_trend()

        # Check if we can generate a learning insight
        if feedback_type == FeedbackType.REJECTED and rejection_reason:
            self._maybe_generate_insight(session_id, rejection_reason, details)

    def get_acceptance_rate(self) -> dict[str, Any]:
        """Get overall and recent acceptance rates.

        This is the headline number: "Your AI output acceptance rate is 84%"
        """
        conn = self._get_conn()

        # Overall
        total = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
        accepted = conn.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE feedback_type IN ('accepted', 'modified')"
        ).fetchone()[0]

        overall_rate = (accepted / total * 100) if total > 0 else 0

        # Last 7 days
        recent = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN feedback_type IN ('accepted', 'modified') THEN 1 ELSE 0 END) as accepted
            FROM feedback_events
            WHERE timestamp >= datetime('now', '-7 days')
            """
        ).fetchone()
        recent_total = recent["total"]
        recent_accepted = recent["accepted"] or 0
        recent_rate = (recent_accepted / recent_total * 100) if recent_total > 0 else 0

        return {
            "overall": {
                "total_sessions": total,
                "acceptance_rate": round(overall_rate, 1),
            },
            "last_7_days": {
                "total_sessions": recent_total,
                "acceptance_rate": round(recent_rate, 1),
            },
            "improvement": round(recent_rate - overall_rate, 1) if total > 10 else None,
        }

    def get_rejection_analysis(self) -> dict[str, Any]:
        """Analyze why outputs are rejected — this drives improvement.

        "43% of rejections are because the agent used wrong patterns.
        Add 2 more contract rules to fix this."
        """
        conn = self._get_conn()
        total_rejections = conn.execute(
            "SELECT COUNT(*) FROM feedback_events WHERE feedback_type = 'rejected'"
        ).fetchone()[0]

        if total_rejections == 0:
            return {"total_rejections": 0, "reasons": []}

        rows = conn.execute(
            """
            SELECT rejection_reason, COUNT(*) as count
            FROM feedback_events
            WHERE feedback_type = 'rejected' AND rejection_reason IS NOT NULL
            GROUP BY rejection_reason
            ORDER BY count DESC
            """
        ).fetchall()

        reasons = []
        for row in rows:
            reason = row["rejection_reason"]
            count = row["count"]
            pct = count / total_rejections * 100

            # Map rejection reasons to actionable suggestions
            suggestion = self._suggestion_for_reason(reason)

            reasons.append({
                "reason": reason,
                "count": count,
                "percentage": round(pct, 1),
                "suggestion": suggestion,
            })

        return {
            "total_rejections": total_rejections,
            "reasons": reasons,
        }

    def get_insights(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get generated learning insights."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM learning_insights ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Get acceptance rate trend over time."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM acceptance_trend ORDER BY date DESC LIMIT ?",
            (days,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_context_effectiveness(self) -> dict[str, Any]:
        """Which context types correlate with acceptance?

        "Sessions with decisions provided had 91% acceptance rate.
        Sessions without decisions had 64% acceptance rate."

        This proves the value of each component.
        """
        conn = self._get_conn()

        # Sessions with decisions
        with_decisions = conn.execute(
            """
            SELECT
                COUNT(DISTINCT cs.session_id) as total,
                SUM(CASE WHEN fe.feedback_type IN ('accepted', 'modified') THEN 1 ELSE 0 END) as accepted
            FROM context_sessions cs
            JOIN feedback_events fe ON cs.session_id = fe.session_id
            WHERE cs.decisions_provided > 0
            """
        ).fetchone()

        # Sessions without decisions
        without_decisions = conn.execute(
            """
            SELECT
                COUNT(DISTINCT cs.session_id) as total,
                SUM(CASE WHEN fe.feedback_type IN ('accepted', 'modified') THEN 1 ELSE 0 END) as accepted
            FROM context_sessions cs
            JOIN feedback_events fe ON cs.session_id = fe.session_id
            WHERE cs.decisions_provided = 0
            """
        ).fetchone()

        # Sessions with contracts
        with_contracts = conn.execute(
            """
            SELECT
                COUNT(DISTINCT cs.session_id) as total,
                SUM(CASE WHEN fe.feedback_type IN ('accepted', 'modified') THEN 1 ELSE 0 END) as accepted
            FROM context_sessions cs
            JOIN feedback_events fe ON cs.session_id = fe.session_id
            WHERE cs.contracts_provided > 0
            """
        ).fetchone()

        def _rate(row: Any) -> float:
            t = row["total"] or 0
            a = row["accepted"] or 0
            return round(a / t * 100, 1) if t > 0 else 0

        return {
            "with_decisions": {
                "sessions": with_decisions["total"] or 0,
                "acceptance_rate": _rate(with_decisions),
            },
            "without_decisions": {
                "sessions": without_decisions["total"] or 0,
                "acceptance_rate": _rate(without_decisions),
            },
            "with_contracts": {
                "sessions": with_contracts["total"] or 0,
                "acceptance_rate": _rate(with_contracts),
            },
            "insight": self._generate_effectiveness_insight(
                with_decisions, without_decisions, with_contracts
            ),
        }

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _maybe_generate_insight(
        self,
        session_id: str,
        reason: RejectionReason,
        details: str | None,
    ) -> None:
        """Check if we have enough data to generate a learning insight."""
        conn = self._get_conn()

        # Count recent rejections for this reason
        count = conn.execute(
            """
            SELECT COUNT(*) FROM feedback_events
            WHERE rejection_reason = ? AND timestamp >= datetime('now', '-30 days')
            """,
            (reason.value,),
        ).fetchone()[0]

        # Need at least 3 instances to generate an insight
        if count >= 3:
            suggestion = self._suggestion_for_reason(reason.value)
            conn.execute(
                """
                INSERT INTO learning_insights (timestamp, insight_type, description, confidence, action_taken)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    "pattern_detected",
                    f"Repeated rejection reason: {reason.value} ({count} times in 30 days). {suggestion}",
                    min(0.9, 0.3 + count * 0.1),
                    None,
                ),
            )
            conn.commit()

    def _suggestion_for_reason(self, reason: str) -> str:
        """Map rejection reasons to actionable improvement suggestions."""
        suggestions = {
            "wrong_pattern": (
                "Add a quality contract with the correct pattern as an example. "
                "The agent needs to see DO/DON'T examples for your project's conventions."
            ),
            "missing_context": (
                "Create a decision record for the missing context. "
                "If it's a dependency, check that the code graph captured the relationship."
            ),
            "violated_decision": (
                "The decision exists but wasn't surfaced. Check the decision's "
                "code_anchors — they may need updating after recent refactors."
            ),
            "hallucinated": (
                "Enable the 'no-hallucinated-imports' AI guardrail contract. "
                "The graph can verify import targets exist before the agent uses them."
            ),
            "over_engineered": (
                "Enable the 'no-over-abstraction' AI guardrail. "
                "Consider adding a contract rule: 'no base classes with single implementation.'"
            ),
            "security_issue": (
                "Add security-focused contract rules. Check the community "
                "contract packs for your framework."
            ),
            "wrong_approach": (
                "Create a decision record documenting the correct approach "
                "and why the alternative was rejected."
            ),
        }
        return suggestions.get(reason, "Review the context assembly for this task type.")

    def _generate_effectiveness_insight(
        self, with_dec: Any, without_dec: Any, with_contracts: Any
    ) -> str:
        """Generate a human-readable insight about context effectiveness."""
        parts = []
        wd_total = with_dec["total"] or 0
        wod_total = without_dec["total"] or 0

        if wd_total >= 5 and wod_total >= 5:
            wd_rate = (with_dec["accepted"] or 0) / wd_total * 100
            wod_rate = (without_dec["accepted"] or 0) / wod_total * 100
            diff = wd_rate - wod_rate
            if diff > 5:
                parts.append(
                    f"Sessions with decisions had {diff:.0f}% higher acceptance rate."
                )

        wc_total = with_contracts["total"] or 0
        if wc_total >= 5:
            wc_rate = (with_contracts["accepted"] or 0) / wc_total * 100
            parts.append(f"Contract-guided sessions: {wc_rate:.0f}% acceptance rate.")

        return " ".join(parts) if parts else "Not enough data yet for insights."

    def _update_acceptance_trend(self) -> None:
        """Refresh today's acceptance trend entry."""
        conn = self._get_conn()
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN feedback_type = 'accepted' THEN 1 ELSE 0 END) as accepted,
                SUM(CASE WHEN feedback_type = 'modified' THEN 1 ELSE 0 END) as modified,
                SUM(CASE WHEN feedback_type = 'rejected' THEN 1 ELSE 0 END) as rejected,
                SUM(CASE WHEN feedback_type = 'partial' THEN 1 ELSE 0 END) as partial
            FROM feedback_events
            WHERE timestamp LIKE ?
            """,
            (f"{today}%",),
        ).fetchone()

        total = row["total"]
        accepted = (row["accepted"] or 0) + (row["modified"] or 0)
        rate = (accepted / total * 100) if total > 0 else 0

        conn.execute(
            """
            INSERT INTO acceptance_trend (date, total_sessions, accepted, modified, rejected, partial, acceptance_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_sessions=excluded.total_sessions,
                accepted=excluded.accepted, modified=excluded.modified,
                rejected=excluded.rejected, partial=excluded.partial,
                acceptance_rate=excluded.acceptance_rate
            """,
            (today, total, row["accepted"] or 0, row["modified"] or 0, row["rejected"] or 0, row["partial"] or 0, round(rate, 1)),
        )
        conn.commit()
