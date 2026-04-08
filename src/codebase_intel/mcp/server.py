"""MCP Server — the agent-facing interface for codebase intelligence.

This is the primary integration point. AI agents connect via MCP and
call tools to get structured context for their tasks.

Design principles:
- Every tool is self-describing (clear name, description, parameter docs)
- Tools return structured JSON (not raw text) for reliable parsing
- Errors are returned as structured objects, never thrown as exceptions
- Response sizes are bounded (no unbounded result sets)
- Tools are stateless per-request (state lives in storage layer)

Edge cases:
- MCP client connects before init: tools return "not initialized" error
  with instructions to run `codebase-intel init`
- Concurrent tool calls: safe because storage uses WAL mode and
  tool handlers are independently scoped
- Large response payload: capped at configurable max size, truncated
  with continuation token (future)
- Client disconnects mid-request: handled by MCP framework
- Invalid parameters: Pydantic validation on inputs, structured error on failure
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from codebase_intel.core.config import ProjectConfig
from codebase_intel.core.types import TokenBudget

logger = logging.getLogger(__name__)


def create_server(project_root: Path | None = None) -> Server:
    """Create and configure the MCP server with all tools.

    The server exposes these tools:
    1. get_context — Main tool: assemble context for a task
    2. query_graph — Query the code graph directly
    3. get_decisions — Get decisions relevant to files/tags
    4. get_contracts — Get applicable quality contracts
    5. check_drift — Run drift detection
    6. impact_analysis — What's affected by changes to these files?
    7. get_status — Health check and component status
    """
    server = Server("codebase-intel")
    _root = project_root or Path.cwd()

    # Lazy initialization — components are created on first use
    _state: dict[str, Any] = {"initialized": False}

    async def _ensure_initialized() -> dict[str, Any]:
        """Initialize components lazily on first tool call.

        Edge case: init might fail partially (graph OK, decisions dir missing).
        We track what's available and what's not, and tools degrade gracefully.
        """
        if _state["initialized"]:
            return _state

        try:
            config = ProjectConfig(project_root=_root)
        except Exception as exc:
            _state["error"] = f"Configuration error: {exc}"
            _state["initialized"] = True
            return _state

        _state["config"] = config

        # Initialize graph — open persistent connection for the server lifetime
        try:
            from codebase_intel.graph.storage import GraphStorage

            if config.graph.db_path.exists():
                import aiosqlite

                db = await aiosqlite.connect(str(config.graph.db_path))
                await db.execute("PRAGMA journal_mode=WAL")
                await db.execute("PRAGMA busy_timeout=5000")
                await db.execute("PRAGMA foreign_keys=ON")
                storage = GraphStorage(db, config.project_root)
                await storage._ensure_schema()

                from codebase_intel.graph.query import GraphQueryEngine

                _state["graph_storage"] = storage
                _state["graph_engine"] = GraphQueryEngine(storage)
                _state["graph_db"] = db
                _state["graph_available"] = True
            else:
                _state["graph_available"] = False
        except Exception as exc:
            logger.warning("Graph initialization failed: %s", exc)
            _state["graph_available"] = False

        # Initialize decisions
        try:
            from codebase_intel.decisions.store import DecisionStore

            store = DecisionStore(config.decisions, config.project_root)
            _state["decisions"] = store
            _state["decisions_available"] = True
        except Exception:
            _state["decisions_available"] = False

        # Initialize contracts
        try:
            from codebase_intel.contracts.registry import ContractRegistry

            registry = ContractRegistry(config.contracts, config.project_root)
            registry.load()
            _state["contracts"] = registry
            _state["contracts_available"] = True
        except Exception:
            _state["contracts_available"] = False

        # Initialize analytics tracker (live efficiency tracking)
        try:
            from codebase_intel.analytics.tracker import AnalyticsTracker

            analytics_db = config.project_root / ".codebase-intel" / "analytics.db"
            _state["analytics"] = AnalyticsTracker(analytics_db)
            _state["analytics_available"] = True
        except Exception:
            _state["analytics_available"] = False

        # Initialize feedback tracker
        try:
            from codebase_intel.analytics.feedback import FeedbackTracker

            feedback_db = config.project_root / ".codebase-intel" / "feedback.db"
            _state["feedback"] = FeedbackTracker(feedback_db)
            _state["feedback_available"] = True
        except Exception:
            _state["feedback_available"] = False

        _state["initialized"] = True
        return _state

    # -------------------------------------------------------------------
    # Tool: get_context
    # -------------------------------------------------------------------

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="get_context",
                description=(
                    "Assemble relevant context for a coding task. Gathers related files, "
                    "architectural decisions, and quality contracts within a token budget. "
                    "This is the main tool — use it before starting work on any task."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What you're trying to do (e.g., 'add rate limiting to payment endpoint')",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths you're working on (relative to project root)",
                        },
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific function/class names you're modifying",
                        },
                        "budget_tokens": {
                            "type": "integer",
                            "description": "Max tokens for context (default: 8000)",
                            "default": 8000,
                        },
                    },
                    "required": ["task"],
                },
            ),
            Tool(
                name="query_graph",
                description=(
                    "Query the semantic code graph. Find dependencies, dependents, "
                    "or run impact analysis for changed files."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query_type": {
                            "type": "string",
                            "enum": ["dependencies", "dependents", "impact", "stats"],
                            "description": "Type of graph query",
                        },
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths to query",
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Symbol name to search for",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max traversal depth (default: 2)",
                            "default": 2,
                        },
                    },
                    "required": ["query_type"],
                },
            ),
            Tool(
                name="get_decisions",
                description=(
                    "Get architectural and business decisions relevant to specific files or tags. "
                    "Includes the rationale, constraints, and alternatives considered."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths to find relevant decisions for",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Decision tags to filter by (e.g., 'architecture', 'security')",
                        },
                        "decision_id": {
                            "type": "string",
                            "description": "Get a specific decision by ID",
                        },
                        "include_stale": {
                            "type": "boolean",
                            "description": "Include stale decisions (default: true)",
                            "default": True,
                        },
                    },
                },
            ),
            Tool(
                name="get_contracts",
                description=(
                    "Get quality contracts applicable to specific files. Returns rules "
                    "the code must follow, including AI-specific anti-pattern checks."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "File paths to get contracts for",
                        },
                        "contract_id": {
                            "type": "string",
                            "description": "Get a specific contract by ID",
                        },
                    },
                },
            ),
            Tool(
                name="check_drift",
                description=(
                    "Run drift detection to find stale decisions, orphaned code anchors, "
                    "and outdated graph data. Use this to verify context freshness."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Check drift for specific files (faster than full check)",
                        },
                        "full": {
                            "type": "boolean",
                            "description": "Run full drift check across all records",
                            "default": False,
                        },
                    },
                },
            ),
            Tool(
                name="impact_analysis",
                description=(
                    "Analyze what's affected by changes to specific files. "
                    "Returns a list of files and functions that depend on the changed code."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "changed_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Files that were changed or will be changed",
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Max dependency depth to check (default: 3)",
                            "default": 3,
                        },
                    },
                    "required": ["changed_files"],
                },
            ),
            Tool(
                name="get_status",
                description="Get health status of all codebase-intel components.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            Tool(
                name="record_feedback",
                description=(
                    "Record whether AI-generated code was accepted, modified, or rejected. "
                    "This powers the feedback loop — over time, the system learns which "
                    "context patterns lead to better output."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "Session ID from a previous get_context call",
                        },
                        "feedback": {
                            "type": "string",
                            "enum": ["accepted", "modified", "rejected", "partial"],
                            "description": "How the generated code was received",
                        },
                        "rejection_reason": {
                            "type": "string",
                            "enum": [
                                "wrong_pattern", "missing_context", "violated_decision",
                                "hallucinated", "over_engineered", "security_issue",
                                "wrong_approach", "other",
                            ],
                            "description": "Why the code was rejected (if feedback is 'rejected')",
                        },
                        "details": {
                            "type": "string",
                            "description": "Free-text details about what went wrong or right",
                        },
                    },
                    "required": ["session_id", "feedback"],
                },
            ),
            Tool(
                name="get_efficiency_report",
                description=(
                    "Get live efficiency metrics — token savings, acceptance rate, "
                    "before/after comparison. Proves the value of codebase-intel over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "include_trend": {
                            "type": "boolean",
                            "description": "Include daily trend data (default: true)",
                            "default": True,
                        },
                    },
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Route tool calls to handlers.

        Every handler follows the same pattern:
        1. Ensure initialized
        2. Validate inputs
        3. Execute (catch exceptions → structured error response)
        4. Return structured JSON
        """
        state = await _ensure_initialized()

        if "error" in state:
            return [TextContent(
                type="text",
                text=json.dumps({"error": state["error"], "suggestion": "Run `codebase-intel init`"}),
            )]

        try:
            if name == "get_context":
                result = await _handle_get_context(state, arguments)
            elif name == "query_graph":
                result = await _handle_query_graph(state, arguments)
            elif name == "get_decisions":
                result = await _handle_get_decisions(state, arguments)
            elif name == "get_contracts":
                result = await _handle_get_contracts(state, arguments)
            elif name == "check_drift":
                result = await _handle_check_drift(state, arguments)
            elif name == "impact_analysis":
                result = await _handle_impact_analysis(state, arguments)
            elif name == "get_status":
                result = await _handle_get_status(state, arguments)
            elif name == "record_feedback":
                result = await _handle_record_feedback(state, arguments)
            elif name == "get_efficiency_report":
                result = await _handle_efficiency_report(state, arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
        except Exception as exc:
            logger.exception("Tool %s failed", name)
            result = {
                "error": str(exc),
                "tool": name,
                "suggestion": "Check logs for details",
            }

        return [TextContent(type="text", text=json.dumps(result, default=str, indent=2))]

    # -------------------------------------------------------------------
    # Tool handlers
    # -------------------------------------------------------------------

    async def _handle_get_context(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle the main get_context tool."""
        from codebase_intel.orchestrator.assembler import ContextAssembler

        config = state["config"]
        file_paths = [Path(f) for f in args.get("files", [])]
        budget = TokenBudget(total=args.get("budget_tokens", config.orchestrator.default_budget_tokens))

        # Build assembler with all available components wired in
        assembler = ContextAssembler(
            config=config.orchestrator,
            graph_engine=state.get("graph_engine"),
            decision_store=state.get("decisions"),
            contract_registry=state.get("contracts"),
            analytics_tracker=state.get("analytics"),
        )

        result = await assembler.assemble(
            task_description=args["task"],
            file_paths=file_paths if file_paths else None,
            symbol_names=args.get("symbols"),
            budget=budget,
        )

        return {
            "items": [
                {
                    "source": item.source,
                    "type": item.item_type,
                    "priority": item.priority.value,
                    "content": item.content,
                    "tokens": item.estimated_tokens,
                    "freshness": item.freshness_score,
                }
                for item in result.items
            ],
            "summary": {
                "total_tokens": result.total_tokens,
                "budget_tokens": result.budget_tokens,
                "items_included": len(result.items),
                "items_dropped": result.dropped_count,
                "truncated": result.truncated,
                "assembly_time_ms": round(result.assembly_time_ms, 1),
            },
            "warnings": result.warnings,
            "conflicts": result.conflicts,
        }

    async def _handle_query_graph(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle graph queries."""
        if not state.get("graph_available"):
            return {
                "error": "Code graph not available",
                "suggestion": "Run `codebase-intel init` then `codebase-intel analyze`",
            }

        query_type = args["query_type"]
        engine = state["graph_engine"]
        storage = state["graph_storage"]

        if query_type == "stats":
            stats = await storage.get_stats()
            return {"stats": stats}

        if query_type == "dependencies":
            if "symbol" in args:
                result = await engine.query_by_symbol(args["symbol"], include_depth=args.get("max_depth", 2))
            elif "files" in args:
                result = await engine.query_by_files(
                    [Path(f) for f in args["files"]],
                    include_depth=args.get("max_depth", 2),
                )
            else:
                return {"error": "Provide 'files' or 'symbol' for dependency queries"}

            return {
                "nodes": [
                    {
                        "name": n.qualified_name,
                        "kind": n.kind.value,
                        "file": str(n.file_path),
                        "lines": f"{n.line_range.start}-{n.line_range.end}" if n.line_range else None,
                        "priority": result.priorities.get(n.node_id, "low"),
                        "reason": result.explanations.get(n.node_id, ""),
                    }
                    for n in result.nodes[:50]
                ],
                "total": len(result.nodes),
                "truncated": result.truncated,
                "warnings": result.warnings,
            }

        if query_type == "dependents":
            if "files" not in args:
                return {"error": "Provide 'files' for dependent queries"}
            result = await engine.query_impact(
                [Path(f) for f in args["files"]],
                max_depth=args.get("max_depth", 2),
            )
            return {
                "nodes": [
                    {
                        "name": n.qualified_name,
                        "kind": n.kind.value,
                        "file": str(n.file_path),
                        "priority": result.priorities.get(n.node_id, "low"),
                        "reason": result.explanations.get(n.node_id, ""),
                    }
                    for n in result.nodes[:50]
                ],
                "total": len(result.nodes),
                "truncated": result.truncated,
                "warnings": result.warnings,
            }

        if query_type == "impact":
            if "files" not in args:
                return {"error": "Provide 'files' for impact analysis"}
            impact_map = await storage.impact_analysis(
                [Path(f) for f in args["files"]],
                max_depth=args.get("max_depth", 3),
            )
            return {
                "impact": {
                    fp: [
                        {"name": n.qualified_name, "kind": n.kind.value, "file": str(n.file_path)}
                        for n in nodes[:20]
                    ]
                    for fp, nodes in impact_map.items()
                },
                "total_affected": sum(len(ns) for ns in impact_map.values()),
            }

        return {"error": f"Unknown query_type: {query_type}"}

    async def _handle_get_decisions(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle decision queries."""
        store = state.get("decisions")
        if not store:
            return {
                "error": "Decision journal not available",
                "suggestion": "Run `codebase-intel init` to create the decisions directory",
            }

        # Specific decision by ID
        if "decision_id" in args:
            record = await store.get(args["decision_id"])
            if record:
                return {"decision": record.to_context_string(verbose=True)}
            return {"error": f"Decision '{args['decision_id']}' not found"}

        # Query by files
        if "files" in args:
            file_paths = {Path(f) for f in args["files"]}
            scored = await store.query_by_files(file_paths)
            return {
                "decisions": [
                    {
                        "id": record.id,
                        "title": record.title,
                        "relevance": round(score, 2),
                        "status": record.status.value,
                        "content": record.to_context_string(verbose=score >= 0.5),
                    }
                    for record, score in scored[:10]
                ],
                "total_found": len(scored),
            }

        # Query by tags
        if "tags" in args:
            records = await store.query_by_tags(args["tags"])
            return {
                "decisions": [
                    {"id": r.id, "title": r.title, "tags": r.tags}
                    for r in records[:20]
                ],
            }

        # Return all
        all_records = await store.load_all()
        return {
            "decisions": [
                {"id": r.id, "title": r.title, "status": r.status.value}
                for r in all_records
            ],
            "total": len(all_records),
        }

    async def _handle_get_contracts(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle contract queries."""
        registry = state.get("contracts")
        if not registry:
            return {"error": "Contract registry not available"}

        if "contract_id" in args:
            contract = registry.get(args["contract_id"])
            if contract:
                return {"contract": contract.to_context_string(verbose=True)}
            return {"error": f"Contract '{args['contract_id']}' not found"}

        if "files" in args:
            all_contracts = []
            for f in args["files"]:
                applicable = registry.get_for_file(Path(f))
                for c in applicable:
                    if c.id not in {ac["id"] for ac in all_contracts}:
                        all_contracts.append({
                            "id": c.id,
                            "name": c.name,
                            "priority": c.priority,
                            "rule_count": len(c.rules),
                            "content": c.to_context_string(verbose=True),
                        })
            return {"contracts": all_contracts}

        # Return all
        all_contracts = registry.get_all()
        return {
            "contracts": [
                {"id": c.id, "name": c.name, "rules": len(c.rules), "builtin": c.is_builtin}
                for c in all_contracts
            ],
        }

    async def _handle_check_drift(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle drift detection."""
        from codebase_intel.drift.detector import DriftDetector

        config = state["config"]
        detector = DriftDetector(
            config=config.drift,
            project_root=config.project_root,
            graph_storage=state.get("graph_storage"),
            decision_store=state.get("decisions"),
        )

        if args.get("full"):
            report = await detector.full_check()
        elif "files" in args:
            report = await detector.check_files([Path(f) for f in args["files"]])
        else:
            report = await detector.full_check()

        return {
            "summary": report.summary,
            "overall_level": report.overall_level.value,
            "items": [
                {
                    "component": item.component,
                    "level": item.level.value,
                    "description": item.description,
                    "remediation": item.remediation,
                }
                for item in report.items[:50]
            ],
            "stats": {
                "graph_stale_files": report.graph_stale_files,
                "decision_stale": report.decision_stale_count,
                "decision_orphaned": report.decision_orphaned_count,
                "rot_detected": report.rot_detected,
                "rot_percentage": round(report.rot_percentage, 2),
            },
        }

    async def _handle_impact_analysis(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle impact analysis."""
        if not state.get("graph_available"):
            return {"error": "Code graph not available", "suggestion": "Run `codebase-intel init`"}

        storage = state["graph_storage"]
        changed = [Path(f) for f in args["changed_files"]]
        max_depth = args.get("max_depth", 3)

        impact_map = await storage.impact_analysis(changed, max_depth=max_depth)

        # Also find relevant decisions for changed files
        decision_context: list[dict[str, Any]] = []
        if state.get("decisions"):
            scored = await state["decisions"].query_by_files(set(changed))
            decision_context = [
                {"id": r.id, "title": r.title, "relevance": round(s, 2)}
                for r, s in scored[:5]
            ]

        return {
            "changed_files": [str(f) for f in changed],
            "impact": {
                fp: [
                    {"name": n.qualified_name, "kind": n.kind.value, "file": str(n.file_path)}
                    for n in nodes[:20]
                ]
                for fp, nodes in impact_map.items()
            },
            "total_affected": sum(len(ns) for ns in impact_map.values()),
            "related_decisions": decision_context,
        }

    async def _handle_get_status(
        state: dict[str, Any], _args: dict[str, Any]
    ) -> dict[str, Any]:
        """Return component health status."""
        config = state.get("config")
        root = str(config.project_root) if config else str(_root)

        status: dict[str, Any] = {
            "project_root": root,
            "components": {
                "graph": "available" if state.get("graph_available") else "not_initialized",
                "decisions": "available" if state.get("decisions_available") else "not_initialized",
                "contracts": "available" if state.get("contracts_available") else "not_initialized",
            },
            "version": "0.1.0",
        }

        # Add graph stats if available
        if state.get("graph_available"):
            storage = state["graph_storage"]
            stats = await storage.get_stats()
            status["graph_stats"] = stats

        # Add decision count if available
        if state.get("decisions"):
            all_decisions = await state["decisions"].load_all()
            status["decision_count"] = len(all_decisions)

        # Add contract count if available
        if state.get("contracts"):
            all_contracts = state["contracts"].get_all()
            status["contract_count"] = len(all_contracts)
            status["builtin_contracts"] = sum(1 for c in all_contracts if c.is_builtin)

        return status

    async def _handle_record_feedback(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Record feedback on AI-generated code quality."""
        feedback_tracker = state.get("feedback")
        if not feedback_tracker:
            return {"error": "Feedback tracking not available"}

        from codebase_intel.analytics.feedback import FeedbackType, RejectionReason

        session_id = args["session_id"]
        feedback_type = FeedbackType(args["feedback"])
        rejection_reason = None
        if args.get("rejection_reason"):
            rejection_reason = RejectionReason(args["rejection_reason"])

        # Start session if it doesn't exist yet
        feedback_tracker.start_session(
            session_id=session_id,
            task_description=args.get("details", ""),
            files_involved=[],
        )

        feedback_tracker.record_feedback(
            session_id=session_id,
            feedback_type=feedback_type,
            rejection_reason=rejection_reason,
            details=args.get("details"),
        )

        # Return current acceptance rate so the agent sees the impact
        acceptance = feedback_tracker.get_acceptance_rate()
        effectiveness = feedback_tracker.get_context_effectiveness()

        response: dict[str, Any] = {
            "recorded": True,
            "session_id": session_id,
            "feedback": feedback_type.value,
            "acceptance_rate": acceptance,
        }

        if rejection_reason:
            analysis = feedback_tracker.get_rejection_analysis()
            response["rejection_analysis"] = analysis
            # Show actionable suggestion
            suggestion = feedback_tracker._suggestion_for_reason(rejection_reason.value)
            response["improvement_suggestion"] = suggestion

        if effectiveness.get("insight"):
            response["effectiveness_insight"] = effectiveness["insight"]

        return response

    async def _handle_efficiency_report(
        state: dict[str, Any], args: dict[str, Any]
    ) -> dict[str, Any]:
        """Return live efficiency metrics — the proof layer."""
        analytics = state.get("analytics")
        feedback = state.get("feedback")

        report: dict[str, Any] = {}

        # Token efficiency from analytics
        if analytics:
            stats = analytics.get_lifetime_stats()
            comparison = analytics.get_before_after_comparison()
            report["token_efficiency"] = stats["tokens"]
            report["context_quality"] = stats["context_quality"]
            report["before_vs_after"] = comparison

            if args.get("include_trend", True):
                report["daily_trend"] = analytics.get_daily_trend(14)

            benchmarks = analytics.get_benchmark_results()
            if benchmarks:
                report["benchmarks"] = [
                    {
                        "project": b["repo_name"],
                        "files": b["total_files"],
                        "reduction_pct": b["avg_token_reduction_pct"],
                        "date": b["timestamp"][:10],
                    }
                    for b in benchmarks[:5]
                ]
        else:
            report["token_efficiency"] = {"message": "No analytics data yet. Run `codebase-intel benchmark` to start."}

        # Acceptance rate from feedback
        if feedback:
            report["acceptance_rate"] = feedback.get_acceptance_rate()
            report["rejection_analysis"] = feedback.get_rejection_analysis()
            report["context_effectiveness"] = feedback.get_context_effectiveness()

            insights = feedback.get_insights(5)
            if insights:
                report["learning_insights"] = [
                    {"description": i["description"], "confidence": i["confidence"]}
                    for i in insights
                ]
        else:
            report["acceptance_rate"] = {"message": "No feedback data yet. Use record_feedback after code reviews."}

        # Summary headline
        total_requests = 0
        tokens_saved = 0
        if analytics:
            s = analytics.get_lifetime_stats()
            total_requests = s["total_requests"]
            tokens_saved = s["tokens"]["total_saved"]

        acceptance_pct = 0.0
        if feedback:
            ar = feedback.get_acceptance_rate()
            acceptance_pct = ar["overall"]["acceptance_rate"]

        report["headline"] = {
            "total_requests": total_requests,
            "tokens_saved": tokens_saved,
            "acceptance_rate_pct": acceptance_pct,
            "message": (
                f"codebase-intel has processed {total_requests} context requests, "
                f"saved {tokens_saved:,} tokens, "
                f"with {acceptance_pct:.0f}% AI output acceptance rate."
                if total_requests > 0
                else "No data yet. Use get_context and record_feedback to start tracking."
            ),
        }

        return report

    return server


async def run_server(project_root: Path | None = None) -> None:
    """Run the MCP server over stdio.

    This is the entry point for `codebase-intel serve`.
    """
    server = create_server(project_root)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
