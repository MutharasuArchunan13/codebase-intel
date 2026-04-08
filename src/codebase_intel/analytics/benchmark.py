"""Benchmark system — measures codebase-intel efficiency against real projects.

Runs three scenarios per test case:
1. NAIVE: Read all files in the scope (what an agent without context does)
2. GRAPH: Use code graph to find relevant files only
3. FULL: Graph + decisions + contracts (the complete pipeline)

For each scenario, measures:
- Token count (via tiktoken)
- Number of files included
- Number of decisions/contracts surfaced
- Assembly time

Produces a reproducible report: run `codebase-intel benchmark` and get the same
numbers every time for the same repo state.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codebase_intel.analytics.tracker import AnalyticsTracker

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkScenario:
    """A single benchmark test case: a task + expected scope."""

    name: str
    task_description: str
    target_files: list[str]  # Relative paths to the files being "edited"
    expected_scope: str  # "narrow" (1-5 files), "medium" (5-20), "wide" (20+)


@dataclass
class ScenarioResult:
    """Result of running one scenario in all three modes."""

    name: str
    task_description: str
    target_files: int

    # Naive: read all files in the target directory
    naive_tokens: int = 0
    naive_files: int = 0

    # Graph: use code graph for relevant files
    graph_tokens: int = 0
    graph_files: int = 0

    # Full: graph + decisions + contracts
    full_tokens: int = 0
    full_files: int = 0
    decisions_surfaced: int = 0
    contracts_applied: int = 0
    drift_warnings: int = 0

    # Timings
    graph_assembly_ms: float = 0.0
    full_assembly_ms: float = 0.0

    @property
    def naive_vs_graph_reduction(self) -> float:
        if self.naive_tokens == 0:
            return 0.0
        return (1 - self.graph_tokens / self.naive_tokens) * 100

    @property
    def naive_vs_full_reduction(self) -> float:
        if self.naive_tokens == 0:
            return 0.0
        return (1 - self.full_tokens / self.naive_tokens) * 100

    @property
    def multiplier(self) -> float:
        if self.full_tokens == 0:
            return 0.0
        return self.naive_tokens / self.full_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task_description[:80],
            "target_files": self.target_files,
            "naive_tokens": self.naive_tokens,
            "naive_files": self.naive_files,
            "graph_tokens": self.graph_tokens,
            "graph_files": self.graph_files,
            "full_tokens": self.full_tokens,
            "full_files": self.full_files,
            "decisions_surfaced": self.decisions_surfaced,
            "contracts_applied": self.contracts_applied,
            "reduction_pct": round(self.naive_vs_full_reduction, 1),
            "multiplier": round(self.multiplier, 1),
            "graph_assembly_ms": round(self.graph_assembly_ms, 1),
            "full_assembly_ms": round(self.full_assembly_ms, 1),
        }


@dataclass
class BenchmarkReport:
    """Complete benchmark report for a project."""

    repo_name: str
    repo_path: str
    total_files: int = 0
    total_nodes: int = 0
    total_edges: int = 0
    build_time_ms: float = 0.0
    scenarios: list[ScenarioResult] = field(default_factory=list)

    @property
    def avg_reduction_pct(self) -> float:
        reductions = [s.naive_vs_full_reduction for s in self.scenarios if s.naive_tokens > 0]
        return sum(reductions) / max(len(reductions), 1)

    @property
    def avg_multiplier(self) -> float:
        mults = [s.multiplier for s in self.scenarios if s.full_tokens > 0]
        return sum(mults) / max(len(mults), 1)

    @property
    def total_decisions_surfaced(self) -> int:
        return sum(s.decisions_surfaced for s in self.scenarios)

    @property
    def total_contracts_applied(self) -> int:
        return sum(s.contracts_applied for s in self.scenarios)

    def format_table(self) -> str:
        """Format as a markdown table for README/reports."""
        lines = [
            f"## Benchmark: {self.repo_name}",
            f"",
            f"**Graph:** {self.total_files} files → {self.total_nodes} nodes, {self.total_edges} edges (built in {self.build_time_ms:.0f}ms)",
            f"",
            f"| Scenario | Naive Tokens | Graph Tokens | Full Tokens | Reduction | Multiplier | Decisions | Contracts |",
            f"|---|---:|---:|---:|---:|---:|---:|---:|",
        ]

        for s in self.scenarios:
            lines.append(
                f"| {s.name} | {s.naive_tokens:,} | {s.graph_tokens:,} | "
                f"{s.full_tokens:,} | {s.naive_vs_full_reduction:.0f}% | "
                f"{s.multiplier:.1f}x | {s.decisions_surfaced} | {s.contracts_applied} |"
            )

        lines.append(
            f"| **Average** | | | | **{self.avg_reduction_pct:.0f}%** | "
            f"**{self.avg_multiplier:.1f}x** | **{self.total_decisions_surfaced}** | "
            f"**{self.total_contracts_applied}** |"
        )

        return "\n".join(lines)


class BenchmarkRunner:
    """Runs benchmarks against a project."""

    def __init__(self, project_root: Path) -> None:
        self._project_root = project_root

    async def run(
        self,
        scenarios: list[BenchmarkScenario] | None = None,
        tracker: AnalyticsTracker | None = None,
    ) -> BenchmarkReport:
        """Run full benchmark suite against the project.

        If no scenarios provided, auto-generates them by picking files
        at different depths and dependency counts.
        """
        from codebase_intel.core.config import ProjectConfig
        from codebase_intel.graph.builder import GraphBuilder
        from codebase_intel.graph.query import GraphQueryEngine
        from codebase_intel.graph.storage import GraphStorage

        config = ProjectConfig(project_root=self._project_root)

        report = BenchmarkReport(
            repo_name=self._project_root.name,
            repo_path=str(self._project_root),
        )

        # Build graph
        start = time.monotonic()
        async with GraphStorage.open(config.graph, self._project_root) as storage:
            builder = GraphBuilder(config, storage)
            build_result = await builder.full_build()

            report.total_files = build_result.processed
            report.total_nodes = build_result.nodes_created
            report.total_edges = build_result.edges_created
            report.build_time_ms = (time.monotonic() - start) * 1000

            engine = GraphQueryEngine(storage)

            # Auto-generate scenarios if none provided
            if scenarios is None:
                scenarios = await self._auto_scenarios(storage)

            # Run each scenario
            for scenario in scenarios:
                result = await self._run_scenario(
                    scenario, config, storage, engine
                )
                report.scenarios.append(result)

        # Record in tracker if provided
        if tracker:
            tracker.record_benchmark(
                repo_name=report.repo_name,
                repo_path=report.repo_path,
                total_files=report.total_files,
                total_nodes=report.total_nodes,
                total_edges=report.total_edges,
                scenarios=[s.to_dict() for s in report.scenarios],
                build_time_ms=report.build_time_ms,
            )

        return report

    async def _run_scenario(
        self,
        scenario: BenchmarkScenario,
        config: Any,
        storage: Any,
        engine: Any,
    ) -> ScenarioResult:
        """Run a single scenario in naive, graph, and full modes."""
        from codebase_intel.orchestrator.assembler import estimate_tokens

        target_paths = [self._project_root / f for f in scenario.target_files]
        existing_paths = [p for p in target_paths if p.exists()]

        if not existing_paths:
            return ScenarioResult(
                name=scenario.name,
                task_description=scenario.task_description,
                target_files=0,
            )

        result = ScenarioResult(
            name=scenario.name,
            task_description=scenario.task_description,
            target_files=len(existing_paths),
        )

        # --- NAIVE: what an agent without context tools typically reads ---
        # Agents usually read the target file + parent dir + any file they
        # can grep for imports. We simulate: all files in parent dir + parent's
        # parent dir (2 levels up from each target file).
        naive_content = ""
        naive_files: set[Path] = set()
        code_extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".rb"}
        for fp in existing_paths:
            for ancestor_dir in [fp.parent, fp.parent.parent]:
                if not ancestor_dir.exists():
                    continue
                try:
                    for child in ancestor_dir.iterdir():
                        if child.is_file() and child.suffix in code_extensions and child not in naive_files:
                            naive_files.add(child)
                            try:
                                naive_content += child.read_text(encoding="utf-8", errors="ignore")
                            except OSError:
                                pass
                except OSError:
                    pass

        result.naive_tokens = estimate_tokens(naive_content)
        result.naive_files = len(naive_files)

        # --- GRAPH: use code graph for relevant files ---
        start = time.monotonic()
        graph_result = await engine.query_by_files(existing_paths, include_depth=2)
        result.graph_assembly_ms = (time.monotonic() - start) * 1000

        graph_content = ""
        graph_files_set: set[Path] = set()
        for node in graph_result.nodes:
            if node.file_path not in graph_files_set and node.file_path.exists():
                graph_files_set.add(node.file_path)
                try:
                    graph_content += node.file_path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    pass

        result.graph_tokens = estimate_tokens(graph_content)
        result.graph_files = len(graph_files_set)

        # --- FULL: graph + decisions + contracts ---
        start = time.monotonic()

        from codebase_intel.contracts.registry import ContractRegistry
        from codebase_intel.decisions.store import DecisionStore
        from codebase_intel.orchestrator.assembler import ContextAssembler
        from codebase_intel.core.types import TokenBudget

        decision_store = DecisionStore(config.decisions, self._project_root)
        contract_registry = ContractRegistry(config.contracts, self._project_root)
        contract_registry.load()

        assembler = ContextAssembler(
            config=config.orchestrator,
            graph_engine=engine,
            decision_store=decision_store,
            contract_registry=contract_registry,
        )

        assembled = await assembler.assemble(
            task_description=scenario.task_description,
            file_paths=existing_paths,
            budget=TokenBudget(total=8_000),  # Realistic agent budget — shows how well we prioritize
        )

        result.full_assembly_ms = (time.monotonic() - start) * 1000
        result.full_tokens = assembled.total_tokens
        result.full_files = len({
            item.metadata.get("file_path")
            for item in assembled.items
            if item.metadata.get("file_path")
        })
        result.decisions_surfaced = sum(
            1 for item in assembled.items if item.item_type == "decision"
        )
        result.contracts_applied = sum(
            1 for item in assembled.items if item.item_type == "contract_rule"
        )
        result.drift_warnings = len(assembled.warnings)

        return result

    async def _auto_scenarios(self, storage: Any) -> list[BenchmarkScenario]:
        """Auto-generate benchmark scenarios from the project structure.

        Picks files at different levels:
        1. A deep leaf file (few dependents) — narrow scope
        2. A mid-level file (some dependents) — medium scope
        3. A core/shared file (many dependents) — wide scope
        """
        # Get files sorted by number of nodes (proxy for complexity)
        cursor = await storage._db.execute(
            """
            SELECT file_path, COUNT(*) as node_count
            FROM nodes
            WHERE is_test = 0 AND is_generated = 0
            GROUP BY file_path
            ORDER BY node_count DESC
            """
        )
        files = await cursor.fetchall()

        if not files:
            return []

        scenarios: list[BenchmarkScenario] = []

        # Core file (most nodes — likely a models or utils file)
        if len(files) >= 1:
            core_file = files[0][0]
            scenarios.append(BenchmarkScenario(
                name="Core module change",
                task_description=f"Refactor the core module at {Path(core_file).name}",
                target_files=[core_file],
                expected_scope="wide",
            ))

        # Mid-level file
        mid_idx = len(files) // 3
        if len(files) > mid_idx:
            mid_file = files[mid_idx][0]
            scenarios.append(BenchmarkScenario(
                name="Feature module change",
                task_description=f"Add a new endpoint to {Path(mid_file).name}",
                target_files=[mid_file],
                expected_scope="medium",
            ))

        # Leaf file (fewest nodes)
        if len(files) >= 3:
            leaf_file = files[-1][0]
            scenarios.append(BenchmarkScenario(
                name="Leaf file change",
                task_description=f"Fix a bug in {Path(leaf_file).name}",
                target_files=[leaf_file],
                expected_scope="narrow",
            ))

        # Multi-file change
        if len(files) >= 5:
            multi_files = [files[0][0], files[len(files) // 2][0]]
            scenarios.append(BenchmarkScenario(
                name="Cross-module refactor",
                task_description="Refactor shared types used across multiple modules",
                target_files=multi_files,
                expected_scope="wide",
            ))

        return scenarios
