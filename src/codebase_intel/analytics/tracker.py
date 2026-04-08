"""Analytics tracker — records every context assembly and measures efficiency.

This is the "proof" layer. Every time an agent asks for context, we record:
- How many tokens the naive approach would use (read all files)
- How many tokens the graph-targeted approach used
- How many tokens the full pipeline used (graph + decisions + contracts)
- How many decisions were surfaced
- How many contract violations were caught
- How many drift warnings were included

Over time, this builds an undeniable case: "codebase-intel saved you X tokens,
caught Y violations, and surfaced Z decisions you would have missed."

Storage: SQLite in .codebase-intel/analytics.db — lightweight, portable, queryable.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ANALYTICS_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    task_description TEXT NOT NULL,
    files_requested INTEGER NOT NULL DEFAULT 0,

    -- Token metrics (the core efficiency proof)
    naive_tokens INTEGER NOT NULL DEFAULT 0,
    graph_tokens INTEGER NOT NULL DEFAULT 0,
    full_tokens INTEGER NOT NULL DEFAULT 0,
    budget_tokens INTEGER NOT NULL DEFAULT 0,

    -- Context composition
    items_included INTEGER NOT NULL DEFAULT 0,
    items_dropped INTEGER NOT NULL DEFAULT 0,
    decisions_surfaced INTEGER NOT NULL DEFAULT 0,
    contracts_applied INTEGER NOT NULL DEFAULT 0,
    drift_warnings INTEGER NOT NULL DEFAULT 0,
    conflicts_detected INTEGER NOT NULL DEFAULT 0,

    -- Quality signals
    truncated INTEGER NOT NULL DEFAULT 0,
    assembly_time_ms REAL NOT NULL DEFAULT 0,

    -- Metadata
    metadata_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS daily_summary (
    date TEXT PRIMARY KEY,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_naive_tokens INTEGER NOT NULL DEFAULT 0,
    total_graph_tokens INTEGER NOT NULL DEFAULT 0,
    total_full_tokens INTEGER NOT NULL DEFAULT 0,
    total_decisions_surfaced INTEGER NOT NULL DEFAULT 0,
    total_contracts_applied INTEGER NOT NULL DEFAULT 0,
    total_drift_warnings INTEGER NOT NULL DEFAULT 0,
    total_violations_caught INTEGER NOT NULL DEFAULT 0,
    avg_token_reduction_pct REAL NOT NULL DEFAULT 0,
    avg_assembly_time_ms REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    total_files INTEGER NOT NULL DEFAULT 0,
    total_nodes INTEGER NOT NULL DEFAULT 0,
    total_edges INTEGER NOT NULL DEFAULT 0,

    -- Benchmark scenarios
    scenarios_json TEXT NOT NULL DEFAULT '[]',

    -- Aggregate results
    avg_naive_tokens INTEGER NOT NULL DEFAULT 0,
    avg_graph_tokens INTEGER NOT NULL DEFAULT 0,
    avg_full_tokens INTEGER NOT NULL DEFAULT 0,
    avg_token_reduction_pct REAL NOT NULL DEFAULT 0,
    decisions_available INTEGER NOT NULL DEFAULT 0,
    contracts_available INTEGER NOT NULL DEFAULT 0,

    build_time_ms REAL NOT NULL DEFAULT 0,
    metadata_json TEXT DEFAULT '{}'
);
"""


class AnalyticsTracker:
    """Records and queries efficiency metrics."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(ANALYTICS_SCHEMA)
        return self._conn

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # -------------------------------------------------------------------
    # Record events
    # -------------------------------------------------------------------

    def record_context_event(
        self,
        task_description: str,
        files_requested: int,
        naive_tokens: int,
        graph_tokens: int,
        full_tokens: int,
        budget_tokens: int,
        items_included: int = 0,
        items_dropped: int = 0,
        decisions_surfaced: int = 0,
        contracts_applied: int = 0,
        drift_warnings: int = 0,
        conflicts_detected: int = 0,
        truncated: bool = False,
        assembly_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Record a single context assembly event.

        Returns the event ID.
        """
        conn = self._get_conn()
        cursor = conn.execute(
            """
            INSERT INTO context_events (
                timestamp, task_description, files_requested,
                naive_tokens, graph_tokens, full_tokens, budget_tokens,
                items_included, items_dropped, decisions_surfaced,
                contracts_applied, drift_warnings, conflicts_detected,
                truncated, assembly_time_ms, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                task_description[:500],
                files_requested,
                naive_tokens,
                graph_tokens,
                full_tokens,
                budget_tokens,
                items_included,
                items_dropped,
                decisions_surfaced,
                contracts_applied,
                drift_warnings,
                conflicts_detected,
                int(truncated),
                assembly_time_ms,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        self._update_daily_summary()
        return cursor.lastrowid or 0

    def record_benchmark(
        self,
        repo_name: str,
        repo_path: str,
        total_files: int,
        total_nodes: int,
        total_edges: int,
        scenarios: list[dict[str, Any]],
        decisions_available: int = 0,
        contracts_available: int = 0,
        build_time_ms: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Record a benchmark run result."""
        conn = self._get_conn()

        # Compute aggregates from scenarios
        naive_tokens = [s.get("naive_tokens", 0) for s in scenarios]
        graph_tokens = [s.get("graph_tokens", 0) for s in scenarios]
        full_tokens = [s.get("full_tokens", 0) for s in scenarios]

        avg_naive = sum(naive_tokens) // max(len(naive_tokens), 1)
        avg_graph = sum(graph_tokens) // max(len(graph_tokens), 1)
        avg_full = sum(full_tokens) // max(len(full_tokens), 1)

        reductions = []
        for n, f in zip(naive_tokens, full_tokens):
            if n > 0:
                reductions.append((1 - f / n) * 100)
        avg_reduction = sum(reductions) / max(len(reductions), 1)

        cursor = conn.execute(
            """
            INSERT INTO benchmark_runs (
                timestamp, repo_name, repo_path,
                total_files, total_nodes, total_edges,
                scenarios_json,
                avg_naive_tokens, avg_graph_tokens, avg_full_tokens,
                avg_token_reduction_pct,
                decisions_available, contracts_available,
                build_time_ms, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(UTC).isoformat(),
                repo_name,
                repo_path,
                total_files,
                total_nodes,
                total_edges,
                json.dumps(scenarios),
                avg_naive,
                avg_graph,
                avg_full,
                round(avg_reduction, 1),
                decisions_available,
                contracts_available,
                build_time_ms,
                json.dumps(metadata or {}),
            ),
        )
        conn.commit()
        return cursor.lastrowid or 0

    # -------------------------------------------------------------------
    # Query metrics
    # -------------------------------------------------------------------

    def get_lifetime_stats(self) -> dict[str, Any]:
        """Get all-time efficiency statistics."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(naive_tokens), 0) as total_naive,
                COALESCE(SUM(full_tokens), 0) as total_full,
                COALESCE(SUM(graph_tokens), 0) as total_graph,
                COALESCE(SUM(decisions_surfaced), 0) as total_decisions,
                COALESCE(SUM(contracts_applied), 0) as total_contracts,
                COALESCE(SUM(drift_warnings), 0) as total_drift,
                COALESCE(AVG(assembly_time_ms), 0) as avg_assembly_ms,
                COALESCE(SUM(conflicts_detected), 0) as total_conflicts,
                COALESCE(SUM(items_dropped), 0) as total_dropped
            FROM context_events
            """
        ).fetchone()

        total_naive = row["total_naive"]
        total_full = row["total_full"]
        tokens_saved = total_naive - total_full
        reduction_pct = (tokens_saved / total_naive * 100) if total_naive > 0 else 0

        return {
            "total_requests": row["total_requests"],
            "tokens": {
                "total_naive": total_naive,
                "total_graph": row["total_graph"],
                "total_full": total_full,
                "total_saved": tokens_saved,
                "reduction_pct": round(reduction_pct, 1),
            },
            "context_quality": {
                "decisions_surfaced": row["total_decisions"],
                "contracts_applied": row["total_contracts"],
                "drift_warnings": row["total_drift"],
                "conflicts_detected": row["total_conflicts"],
            },
            "performance": {
                "avg_assembly_ms": round(row["avg_assembly_ms"], 1),
                "items_dropped": row["total_dropped"],
            },
        }

    def get_daily_trend(self, days: int = 30) -> list[dict[str, Any]]:
        """Get daily metrics for the last N days."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM daily_summary
            ORDER BY date DESC
            LIMIT ?
            """,
            (days,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_recent_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get most recent context events."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM context_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_benchmark_results(self) -> list[dict[str, Any]]:
        """Get all benchmark run results."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT * FROM benchmark_runs
            ORDER BY timestamp DESC
            """,
        ).fetchall()
        return [dict(row) for row in rows]

    def get_before_after_comparison(self) -> dict[str, Any]:
        """Generate a before/after comparison summary.

        "Before" = naive approach (read all requested files)
        "After" = codebase-intel (graph + decisions + contracts)

        This is the money chart for the README and dashboard.
        """
        stats = self.get_lifetime_stats()
        total_requests = stats["total_requests"]

        if total_requests == 0:
            return {
                "has_data": False,
                "message": "No context events recorded yet. Use the MCP server to start tracking.",
            }

        tokens = stats["tokens"]
        quality = stats["context_quality"]

        return {
            "has_data": True,
            "requests_analyzed": total_requests,
            "before": {
                "label": "Without codebase-intel",
                "tokens_per_request": tokens["total_naive"] // max(total_requests, 1),
                "decisions_available": 0,
                "contract_checks": 0,
                "drift_awareness": False,
                "knows_why": False,
            },
            "after": {
                "label": "With codebase-intel",
                "tokens_per_request": tokens["total_full"] // max(total_requests, 1),
                "decisions_available": quality["decisions_surfaced"],
                "contract_checks": quality["contracts_applied"],
                "drift_awareness": True,
                "knows_why": True,
            },
            "improvement": {
                "token_reduction_pct": tokens["reduction_pct"],
                "tokens_saved_total": tokens["total_saved"],
                "multiplier": round(
                    tokens["total_naive"] / max(tokens["total_full"], 1), 1
                ),
                "decisions_that_prevented_mistakes": quality["decisions_surfaced"],
                "violations_caught_before_generation": quality["contracts_applied"],
            },
        }

    # -------------------------------------------------------------------
    # Internal
    # -------------------------------------------------------------------

    def _update_daily_summary(self) -> None:
        """Refresh today's daily summary row."""
        conn = self._get_conn()
        today = datetime.now(UTC).strftime("%Y-%m-%d")

        row = conn.execute(
            """
            SELECT
                COUNT(*) as total_requests,
                COALESCE(SUM(naive_tokens), 0) as total_naive,
                COALESCE(SUM(graph_tokens), 0) as total_graph,
                COALESCE(SUM(full_tokens), 0) as total_full,
                COALESCE(SUM(decisions_surfaced), 0) as total_decisions,
                COALESCE(SUM(contracts_applied), 0) as total_contracts,
                COALESCE(SUM(drift_warnings), 0) as total_drift,
                COALESCE(AVG(assembly_time_ms), 0) as avg_assembly_ms
            FROM context_events
            WHERE timestamp LIKE ?
            """,
            (f"{today}%",),
        ).fetchone()

        total_naive = row["total_naive"]
        total_full = row["total_full"]
        reduction = (1 - total_full / total_naive) * 100 if total_naive > 0 else 0

        conn.execute(
            """
            INSERT INTO daily_summary (
                date, total_requests, total_naive_tokens, total_graph_tokens,
                total_full_tokens, total_decisions_surfaced, total_contracts_applied,
                total_drift_warnings, avg_token_reduction_pct, avg_assembly_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                total_requests=excluded.total_requests,
                total_naive_tokens=excluded.total_naive_tokens,
                total_graph_tokens=excluded.total_graph_tokens,
                total_full_tokens=excluded.total_full_tokens,
                total_decisions_surfaced=excluded.total_decisions_surfaced,
                total_contracts_applied=excluded.total_contracts_applied,
                total_drift_warnings=excluded.total_drift_warnings,
                avg_token_reduction_pct=excluded.avg_token_reduction_pct,
                avg_assembly_time_ms=excluded.avg_assembly_time_ms
            """,
            (
                today,
                row["total_requests"],
                total_naive,
                row["total_graph"],
                total_full,
                row["total_decisions"],
                row["total_contracts"],
                row["total_drift"],
                round(reduction, 1),
                round(row["avg_assembly_ms"], 1),
            ),
        )
        conn.commit()
