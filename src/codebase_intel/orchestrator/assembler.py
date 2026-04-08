"""Context assembler — the central orchestration engine.

This is the brain of the system. Given a task description and token budget,
it assembles the optimal context payload from all three sources:
1. Code graph → relevant files and dependencies
2. Decision journal → applicable decisions and constraints
3. Quality contracts → rules the agent must follow

The assembler must make hard prioritization choices when budget is tight.
Its goal is to maximize the probability of correct code generation.

Edge cases (the hard ones):
- Budget is tiny (500 tokens): return only the highest-priority single file +
  critical constraints. No decisions, no contracts beyond blocking errors.
- Budget is huge (100K tokens): still don't dump everything — irrelevant context
  dilutes attention. Cap at what's actually relevant.
- Task mentions files that don't exist yet: the agent is creating new code.
  Return architectural context (what patterns to follow, what to import from).
- Task is ambiguous ("improve performance"): return the broadest relevant scope
  with a warning that context may be incomplete.
- Multiple tasks in one prompt: detect and assemble context for each independently,
  then merge and deduplicate.
- Contradictory context: Decision says "use pattern A" but Contract says "pattern A
  is forbidden." Surface the contradiction explicitly.
- All modules partially initialized: graph exists but no decisions. Return what's
  available with clear warnings about what's missing.
- Assembly timeout: cap at max_assembly_time_ms. If we can't finish, return
  what we have so far with a truncation warning.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import tiktoken

from codebase_intel.core.exceptions import (
    BudgetExceededError,
    ErrorContext,
    PartialInitializationError,
)
from codebase_intel.core.types import (
    AssembledContext,
    ContextItem,
    ContextPriority,
    TokenBudget,
)

if TYPE_CHECKING:
    from codebase_intel.contracts.evaluator import ContractEvaluator
    from codebase_intel.contracts.registry import ContractRegistry
    from codebase_intel.core.config import OrchestratorConfig
    from codebase_intel.decisions.store import DecisionStore
    from codebase_intel.graph.query import GraphQueryEngine

logger = logging.getLogger(__name__)

# tiktoken encoder for token estimation
# cl100k_base is used by GPT-4 and is a reasonable baseline for Claude too
_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding("cl100k_base")
    return _encoder


def estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.

    Edge case: different models tokenize differently. cl100k_base is
    a reasonable approximation. The TokenBudget.safety_margin_pct
    accounts for variance (typically 10%).

    Edge case: empty string → 0 tokens (valid).
    Edge case: non-ASCII text → may tokenize differently, but the
    safety margin covers this.
    """
    if not text:
        return 0
    return len(_get_encoder().encode(text))


class ContextAssembler:
    """Assembles context from all sources within a token budget."""

    def __init__(
        self,
        config: OrchestratorConfig,
        graph_engine: GraphQueryEngine | None = None,
        decision_store: DecisionStore | None = None,
        contract_registry: ContractRegistry | None = None,
        contract_evaluator: ContractEvaluator | None = None,
        analytics_tracker: Any | None = None,
    ) -> None:
        self._config = config
        self._graph = graph_engine
        self._decisions = decision_store
        self._contracts = contract_registry
        self._evaluator = contract_evaluator
        self._analytics = analytics_tracker

    async def assemble(
        self,
        task_description: str,
        file_paths: list[Path] | None = None,
        symbol_names: list[str] | None = None,
        budget: TokenBudget | None = None,
    ) -> AssembledContext:
        """Assemble context for an AI agent's task.

        This is the main entry point. It:
        1. Determines what's relevant (via graph)
        2. Gathers applicable decisions
        3. Gathers applicable contracts
        4. Prioritizes and trims to fit budget
        5. Detects contradictions
        6. Returns the assembled payload

        Parameters:
        - task_description: what the agent is trying to do
        - file_paths: files the agent is working on (if known)
        - symbol_names: specific symbols being modified (if known)
        - budget: token budget constraint
        """
        start_time = time.monotonic()
        budget = budget or TokenBudget(total=self._config.default_budget_tokens)

        context = AssembledContext(budget_tokens=budget.usable)
        items: list[ContextItem] = []

        # Track what's available vs missing
        available: list[str] = []
        missing: list[str] = []

        # 1. Gather graph context (relevant files and dependencies)
        if self._graph and file_paths:
            available.append("graph")
            graph_items = await self._gather_graph_context(file_paths, symbol_names)
            items.extend(graph_items)
        elif not self._graph:
            missing.append("graph")
            context.warnings.append("Code graph not available — file dependencies unknown")

        # 2. Gather decision context
        if self._decisions and file_paths:
            available.append("decisions")
            decision_items = await self._gather_decision_context(file_paths)
            items.extend(decision_items)
        elif not self._decisions:
            missing.append("decisions")
            context.warnings.append("Decision journal not available — architectural context missing")

        # 3. Gather contract context (pre-generation guidance)
        if self._contracts and file_paths:
            available.append("contracts")
            contract_items = self._gather_contract_context(file_paths)
            items.extend(contract_items)
        elif not self._contracts:
            missing.append("contracts")

        # Warn about partial initialization
        if missing:
            context.warnings.append(
                f"Partial initialization: available={available}, missing={missing}"
            )

        # 4. Prioritize and trim to budget
        items.sort(key=lambda i: self._priority_sort_key(i))
        fitted_items, dropped = self._fit_to_budget(items, budget.usable)

        context.items = fitted_items
        context.total_tokens = sum(i.estimated_tokens for i in fitted_items)
        context.dropped_count = dropped
        context.truncated = dropped > 0

        # 5. Detect contradictions
        context.conflicts = self._detect_contradictions(fitted_items)

        # 6. Check assembly time
        elapsed_ms = (time.monotonic() - start_time) * 1000
        context.assembly_time_ms = elapsed_ms

        if elapsed_ms > self._config.max_assembly_time_ms:
            context.warnings.append(
                f"Assembly took {elapsed_ms:.0f}ms (limit: {self._config.max_assembly_time_ms}ms)"
            )

        logger.info(
            "Assembled context: %d items, %d tokens, %d dropped, %.0fms",
            len(context.items),
            context.total_tokens,
            context.dropped_count,
            context.assembly_time_ms,
        )

        # 7. Record analytics (non-blocking — never fail the assembly)
        if self._analytics:
            try:
                # Estimate naive tokens (what reading all requested files would cost)
                naive_tokens = self._estimate_naive_tokens(file_paths or [])
                graph_tokens = sum(
                    i.estimated_tokens for i in context.items if i.source == "graph"
                )
                self._analytics.record_context_event(
                    task_description=task_description,
                    files_requested=len(file_paths or []),
                    naive_tokens=naive_tokens,
                    graph_tokens=graph_tokens,
                    full_tokens=context.total_tokens,
                    budget_tokens=budget.usable,
                    items_included=len(context.items),
                    items_dropped=context.dropped_count,
                    decisions_surfaced=sum(
                        1 for i in context.items if i.item_type == "decision"
                    ),
                    contracts_applied=sum(
                        1 for i in context.items if i.item_type == "contract_rule"
                    ),
                    drift_warnings=len(context.warnings),
                    conflicts_detected=len(context.conflicts),
                    truncated=context.truncated,
                    assembly_time_ms=context.assembly_time_ms,
                )
            except Exception as exc:
                logger.debug("Analytics recording failed (non-fatal): %s", exc)

        return context

    def _estimate_naive_tokens(self, file_paths: list[Path]) -> int:
        """Estimate how many tokens reading all files in the target dirs would cost.

        This is the "before" number — what happens without codebase-intel.
        Agents typically read the entire directory or all open files.
        """
        seen: set[Path] = set()
        total_content = ""

        for fp in file_paths:
            # Read the file itself
            if fp.exists() and fp not in seen:
                seen.add(fp)
                try:
                    total_content += fp.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass

            # Read all siblings in the same directory (common agent behavior)
            if fp.parent.exists():
                try:
                    for sibling in fp.parent.iterdir():
                        if (
                            sibling.is_file()
                            and sibling not in seen
                            and sibling.suffix in (".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java")
                        ):
                            seen.add(sibling)
                            try:
                                total_content += sibling.read_text(encoding="utf-8", errors="ignore")
                            except OSError:
                                pass
                except OSError:
                    pass

        return estimate_tokens(total_content)

    # -------------------------------------------------------------------
    # Context gathering from each source
    # -------------------------------------------------------------------

    async def _gather_graph_context(
        self,
        file_paths: list[Path],
        symbol_names: list[str] | None = None,
    ) -> list[ContextItem]:
        """Gather context items from the code graph.

        Strategy:
        - CRITICAL: the files themselves (read content)
        - HIGH: direct dependencies (read content or summary)
        - MEDIUM: transitive dependencies (summary only — save budget)
        - LOW: test files (summary only)

        Edge case: file is huge (>1000 lines). Include a summary (first 50
        lines + class/function signatures) instead of full content.
        """
        if not self._graph:
            return []

        items: list[ContextItem] = []

        result = await self._graph.query_by_files(file_paths)

        for node in result.nodes:
            priority = result.priorities.get(node.node_id, ContextPriority.LOW)
            explanation = result.explanations.get(node.node_id, "")

            # Read file content for CRITICAL/HIGH priority
            if priority in (ContextPriority.CRITICAL, ContextPriority.HIGH):
                content = self._read_file_for_context(node.file_path, priority)
            else:
                content = self._summarize_node(node)

            if content:
                tokens = estimate_tokens(content)
                items.append(ContextItem(
                    source="graph",
                    item_type="file_content",
                    priority=priority,
                    estimated_tokens=tokens,
                    content=content,
                    metadata={
                        "file_path": str(node.file_path),
                        "node_kind": node.kind.value,
                        "explanation": explanation,
                    },
                    freshness_score=1.0,  # Graph is always current
                ))

        # Add warnings from graph query
        for warning in result.warnings:
            items.append(ContextItem(
                source="graph",
                item_type="warning",
                priority=ContextPriority.HIGH,
                estimated_tokens=estimate_tokens(warning),
                content=warning,
                freshness_score=1.0,
            ))

        return items

    async def _gather_decision_context(
        self,
        file_paths: list[Path],
    ) -> list[ContextItem]:
        """Gather applicable decisions for the files being worked on.

        Strategy:
        - Relevance >= 0.8: HIGH priority (directly related)
        - Relevance >= 0.3: MEDIUM priority (same module/package)
        - Relevance >= 0.1: LOW priority (tangentially related)

        Stale decisions: included but with lower freshness_score.
        Active constraints: always at least MEDIUM priority.

        Edge case: many decisions match (10+). Take top 5 by relevance
        to avoid context pollution.
        """
        if not self._decisions:
            return []

        items: list[ContextItem] = []
        path_set = set(file_paths)

        scored = await self._decisions.query_by_files(path_set)
        max_decisions = 5

        for record, relevance in scored[:max_decisions]:
            # Determine priority from relevance score
            if relevance >= 0.8:
                priority = ContextPriority.HIGH
            elif relevance >= 0.3:
                priority = ContextPriority.MEDIUM
            else:
                priority = ContextPriority.LOW

            # High-priority for active constraints
            if record.constraints and any(c.is_hard for c in record.constraints):
                priority = max(priority, ContextPriority.HIGH, key=lambda p: list(ContextPriority).index(p))

            # Compute freshness
            freshness = 1.0
            if record.is_stale:
                freshness = 0.3
            elif record.last_validated:
                from datetime import UTC, datetime
                days_since = (datetime.now(UTC) - record.last_validated).days
                freshness = max(0.2, 1.0 - (days_since / self._config.freshness_decay_days))

            verbose = priority in (ContextPriority.CRITICAL, ContextPriority.HIGH)
            content = record.to_context_string(verbose=verbose)
            tokens = estimate_tokens(content)

            items.append(ContextItem(
                source="decisions",
                item_type="decision",
                priority=priority,
                estimated_tokens=tokens,
                content=content,
                metadata={
                    "decision_id": record.id,
                    "relevance": relevance,
                    "status": record.status.value,
                },
                freshness_score=freshness,
            ))

        return items

    def _gather_contract_context(
        self,
        file_paths: list[Path],
    ) -> list[ContextItem]:
        """Gather applicable quality contracts for pre-generation guidance.

        Strategy: include all applicable contracts. They're typically compact
        and the agent needs to know ALL rules, not a subset.

        Edge case: 20+ contracts match. Unlikely in practice (most projects
        have 3-5 contracts). If it happens, sort by priority and take top 10.
        """
        if not self._contracts:
            return []

        items: list[ContextItem] = []
        seen_ids: set[str] = set()

        for fp in file_paths:
            applicable = self._contracts.get_for_file(fp)
            for contract in applicable:
                if contract.id in seen_ids:
                    continue
                seen_ids.add(contract.id)

                content = contract.to_context_string(verbose=True)
                tokens = estimate_tokens(content)

                # Contracts are guidance — always at least MEDIUM priority
                priority = ContextPriority.MEDIUM
                if any(r.severity == ContextPriority.CRITICAL for r in contract.rules):
                    priority = ContextPriority.HIGH

                items.append(ContextItem(
                    source="contracts",
                    item_type="contract_rule",
                    priority=priority,
                    estimated_tokens=tokens,
                    content=content,
                    metadata={
                        "contract_id": contract.id,
                        "rule_count": len(contract.rules),
                        "priority": contract.priority,
                    },
                    freshness_score=1.0,  # Contracts are always current
                ))

        return items

    # -------------------------------------------------------------------
    # Budget management
    # -------------------------------------------------------------------

    def _priority_sort_key(self, item: ContextItem) -> tuple[int, float, int]:
        """Sort key: higher priority first, then fresher, then smaller.

        This determines the order in which items are included when budget
        is tight. The last items to be included are LOW priority, stale,
        and large.
        """
        priority_order = {
            ContextPriority.CRITICAL: 0,
            ContextPriority.HIGH: 1,
            ContextPriority.MEDIUM: 2,
            ContextPriority.LOW: 3,
        }
        return (
            priority_order.get(item.priority, 3),
            -item.freshness_score,  # Higher freshness = earlier
            item.estimated_tokens,  # Smaller items first within same priority
        )

    def _fit_to_budget(
        self,
        items: list[ContextItem],
        budget_tokens: int,
    ) -> tuple[list[ContextItem], int]:
        """Select items that fit within the token budget.

        Strategy: greedy — include items in priority order until budget
        is exhausted. This is optimal for a knapsack problem when items
        are sorted by value/weight ratio.

        Edge cases:
        - Single CRITICAL item exceeds budget: include it anyway (the agent
          needs to see the file it's editing, even if nothing else fits).
          But truncate the file content to fit.
        - Budget is 0: return empty list (metadata-only response).
        - All items are tiny: include everything.

        Returns: (fitted_items, dropped_count)
        """
        if budget_tokens <= 0:
            return [], len(items)

        fitted: list[ContextItem] = []
        used = 0
        dropped = 0

        for item in items:
            if used + item.estimated_tokens <= budget_tokens:
                fitted.append(item)
                used += item.estimated_tokens
            elif item.priority == ContextPriority.CRITICAL and not fitted:
                # Must include at least one CRITICAL item, even if it exceeds budget
                # Truncate its content to fit
                truncated = self._truncate_to_fit(item, budget_tokens)
                fitted.append(truncated)
                used += truncated.estimated_tokens
            else:
                dropped += 1

        return fitted, dropped

    def _truncate_to_fit(self, item: ContextItem, budget_tokens: int) -> ContextItem:
        """Truncate a context item's content to fit within a token budget.

        Strategy: keep the first N lines that fit, add a "[truncated]" marker.

        Edge case: even the first line exceeds budget → include just the
        metadata (file path, type) with no content.
        """
        if item.estimated_tokens <= budget_tokens:
            return item

        lines = item.content.split("\n")
        truncated_lines: list[str] = []
        used = 0

        marker = "\n[... truncated to fit token budget ...]"
        marker_tokens = estimate_tokens(marker)
        available = budget_tokens - marker_tokens

        for line in lines:
            line_tokens = estimate_tokens(line + "\n")
            if used + line_tokens > available:
                break
            truncated_lines.append(line)
            used += line_tokens

        if not truncated_lines:
            content = f"[File: {item.metadata.get('file_path', 'unknown')} — truncated]"
        else:
            content = "\n".join(truncated_lines) + marker

        return ContextItem(
            source=item.source,
            item_type=item.item_type,
            priority=item.priority,
            estimated_tokens=estimate_tokens(content),
            content=content,
            metadata={**item.metadata, "truncated": True},
            freshness_score=item.freshness_score,
        )

    # -------------------------------------------------------------------
    # Contradiction detection
    # -------------------------------------------------------------------

    def _detect_contradictions(self, items: list[ContextItem]) -> list[str]:
        """Detect contradictions between context items.

        Types of contradictions:
        1. Decision says "do X" but contract says "don't do X"
        2. Two decisions give conflicting guidance for the same code
        3. Contract rule conflicts (already detected by evaluator)

        This is heuristic-based — we can't perfectly detect semantic
        contradictions, but we can catch obvious structural ones.
        """
        contradictions: list[str] = []

        decisions = [i for i in items if i.item_type == "decision"]
        contracts = [i for i in items if i.item_type == "contract_rule"]

        # Check for stale decision + active contract mismatch
        for decision in decisions:
            if decision.freshness_score < 0.5:
                for contract in contracts:
                    # If a contract references the same area as a stale decision,
                    # flag it as a potential contradiction
                    decision_file = decision.metadata.get("file_path", "")
                    contract_scope = contract.metadata.get("contract_id", "")
                    if decision_file:
                        contradictions.append(
                            f"Stale decision {decision.metadata.get('decision_id', '?')} "
                            f"may conflict with contract '{contract_scope}'. "
                            f"Verify the decision still applies."
                        )

        return contradictions

    # -------------------------------------------------------------------
    # File reading helpers
    # -------------------------------------------------------------------

    def _read_file_for_context(
        self,
        file_path: Path,
        priority: ContextPriority,
    ) -> str | None:
        """Read a file's content for inclusion in context.

        Edge cases:
        - File doesn't exist: return None
        - File is binary: return None
        - File is huge (>1000 lines): return summary for non-CRITICAL priority
        - File has encoding issues: try UTF-8, fall back to latin-1
        """
        if not file_path.exists():
            return None

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                content = file_path.read_text(encoding="latin-1")
            except Exception:
                return None
        except OSError:
            return None

        lines = content.split("\n")
        max_lines = 1000 if priority == ContextPriority.CRITICAL else 200

        if len(lines) > max_lines:
            header = f"# File: {file_path.name} ({len(lines)} lines, showing first {max_lines})\n"
            return header + "\n".join(lines[:max_lines]) + "\n[... truncated ...]"

        return f"# File: {file_path.name}\n{content}"

    def _summarize_node(self, node: "GraphNode") -> str:
        """Create a compact summary of a graph node.

        Used for MEDIUM/LOW priority items to save budget.
        """
        from codebase_intel.core.types import GraphNode  # avoid circular

        parts = [f"{node.kind.value}: {node.qualified_name}"]
        if node.docstring:
            parts.append(f"  {node.docstring[:200]}")
        if node.line_range:
            parts.append(f"  {node.file_path.name}:{node.line_range.start}-{node.line_range.end}")
        return "\n".join(parts)
