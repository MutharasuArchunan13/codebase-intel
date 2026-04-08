"""CLI entry point — the developer-facing interface.

Commands:
- init: Initialize codebase-intel for a project
- analyze: Build or update the code graph
- query: Query the graph, decisions, or contracts
- drift: Run drift detection
- mine: Mine git history for decision candidates
- serve: Start the MCP server
- status: Show component health

Design:
- Uses Typer for declarative CLI definition
- Rich for pretty output (tables, panels, progress bars)
- All heavy operations are async (run via asyncio.run)
- Commands degrade gracefully if components aren't initialized
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="codebase-intel",
    help="Codebase Intelligence Platform — structured context for AI coding agents",
    no_args_is_help=True,
)
console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


# -------------------------------------------------------------------
# init
# -------------------------------------------------------------------


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(help="Project root directory"),
    ] = Path("."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Initialize codebase-intel for a project.

    Creates the .codebase-intel directory, builds the initial code graph,
    and generates starter configuration and contract templates.
    """
    _setup_logging(verbose)
    project_root = path.resolve()

    if not project_root.is_dir():
        console.print(f"[red]Not a directory: {project_root}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"Initializing codebase-intel for [bold]{project_root.name}[/bold]",
        title="codebase-intel init",
    ))

    asyncio.run(_init_async(project_root, verbose))


async def _init_async(project_root: Path, verbose: bool) -> None:
    from codebase_intel.core.config import ProjectConfig

    # Create config
    config = ProjectConfig(project_root=project_root)
    config.ensure_dirs()

    # Save default config
    import yaml
    config_file = project_root / ".codebase-intel" / "config.yaml"
    if not config_file.exists():
        config_file.write_text(
            yaml.dump(config.to_yaml_dict(), default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        console.print(f"  Created config: {config_file.relative_to(project_root)}")

    # Create starter contract template
    from codebase_intel.contracts.registry import ContractRegistry
    registry = ContractRegistry(config.contracts, project_root)
    template_path = await registry.create_template(
        "project-rules", f"{project_root.name} Quality Rules"
    )
    console.print(f"  Created contract template: {template_path.relative_to(project_root)}")

    # Build initial graph
    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building code graph...", total=None)

        from codebase_intel.graph.storage import GraphStorage
        from codebase_intel.graph.builder import GraphBuilder

        async with GraphStorage.open(config.graph, project_root) as storage:
            builder = GraphBuilder(config, storage)
            build_progress = await builder.full_build()

        progress.update(task, description="Done!", completed=True)

    # Summary
    table = Table(title="Initialization Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files processed", str(build_progress.processed))
    table.add_row("Files skipped", str(build_progress.skipped))
    table.add_row("Nodes created", str(build_progress.nodes_created))
    table.add_row("Edges created", str(build_progress.edges_created))
    table.add_row("Warnings", str(len(build_progress.warnings)))
    console.print(table)

    if build_progress.warnings:
        console.print(f"\n[yellow]Warnings ({len(build_progress.warnings)}):[/yellow]")
        for w in build_progress.warnings[:10]:
            console.print(f"  {w}")
        if len(build_progress.warnings) > 10:
            console.print(f"  ... and {len(build_progress.warnings) - 10} more")

    # Mine git history for decisions
    try:
        from codebase_intel.decisions.miner import GitMiner
        miner = GitMiner(config.decisions, project_root)
        candidates = await miner.mine_commits(max_commits=200)
        if candidates:
            console.print(
                f"\n[cyan]Found {len(candidates)} decision candidates in git history.[/cyan]"
            )
            console.print("  Run `codebase-intel mine` to review and save them.")
    except Exception:
        pass  # Git mining is best-effort

    console.print(
        "\n[green]Initialization complete![/green] "
        "The MCP server is ready: `codebase-intel serve`"
    )


# -------------------------------------------------------------------
# analyze
# -------------------------------------------------------------------


@app.command()
def analyze(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Only re-parse changed files"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build or update the code graph."""
    _setup_logging(verbose)
    asyncio.run(_analyze_async(path.resolve(), incremental))


async def _analyze_async(project_root: Path, incremental: bool) -> None:
    from codebase_intel.core.config import ProjectConfig
    from codebase_intel.graph.builder import GraphBuilder
    from codebase_intel.graph.storage import GraphStorage

    config = ProjectConfig(project_root=project_root)

    if not config.graph.db_path.parent.exists():
        console.print("[red]Not initialized. Run `codebase-intel init` first.[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        mode = "incremental" if incremental else "full"
        task = progress.add_task(f"Running {mode} analysis...", total=None)

        async with GraphStorage.open(config.graph, project_root) as storage:
            builder = GraphBuilder(config, storage)
            if incremental:
                result = await builder.incremental_build()
            else:
                result = await builder.full_build()

        progress.update(task, description="Done!", completed=True)

    console.print(
        f"[green]Analysis complete:[/green] "
        f"{result.processed} files, {result.nodes_created} nodes, "
        f"{result.edges_created} edges"
    )


# -------------------------------------------------------------------
# drift
# -------------------------------------------------------------------


@app.command()
def drift(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run drift detection across all components."""
    _setup_logging(verbose)
    asyncio.run(_drift_async(path.resolve()))


async def _drift_async(project_root: Path) -> None:
    from codebase_intel.core.config import ProjectConfig
    from codebase_intel.decisions.store import DecisionStore
    from codebase_intel.drift.detector import DriftDetector

    config = ProjectConfig(project_root=project_root)
    decision_store = DecisionStore(config.decisions, project_root)

    detector = DriftDetector(
        config=config.drift,
        project_root=project_root,
        decision_store=decision_store,
    )

    report = await detector.full_check()

    # Display results
    level_colors = {
        "none": "green",
        "low": "cyan",
        "medium": "yellow",
        "high": "red",
        "critical": "bold red",
    }

    color = level_colors.get(report.overall_level.value, "white")
    console.print(Panel(
        f"Overall: [{color}]{report.overall_level.value.upper()}[/{color}]\n{report.summary}",
        title="Drift Report",
    ))

    if report.items:
        table = Table(title="Drift Items")
        table.add_column("Level", style="bold")
        table.add_column("Component")
        table.add_column("Description")
        table.add_column("Fix")

        for item in report.items[:20]:
            level_color = level_colors.get(item.level.value, "white")
            table.add_row(
                f"[{level_color}]{item.level.value}[/{level_color}]",
                item.component,
                item.description[:80],
                item.remediation[:50] if item.remediation else "",
            )
        console.print(table)

        if len(report.items) > 20:
            console.print(f"\n... and {len(report.items) - 20} more items")

    if report.rot_detected:
        console.print(
            f"\n[bold red]CONTEXT ROT DETECTED:[/bold red] "
            f"{report.rot_percentage:.0%} of records are stale. "
            f"Consider running a team review session."
        )


# -------------------------------------------------------------------
# mine
# -------------------------------------------------------------------


@app.command()
def mine(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    max_commits: int = typer.Option(200, "--max", "-m", help="Max commits to scan"),
    save: bool = typer.Option(False, "--save", "-s", help="Auto-save candidates as draft decisions"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Mine git history for decision candidates."""
    _setup_logging(verbose)
    asyncio.run(_mine_async(path.resolve(), max_commits, save))


async def _mine_async(project_root: Path, max_commits: int, save: bool) -> None:
    from codebase_intel.core.config import ProjectConfig
    from codebase_intel.decisions.miner import GitMiner
    from codebase_intel.decisions.store import DecisionStore

    config = ProjectConfig(project_root=project_root)
    miner = GitMiner(config.decisions, project_root)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Mining last {max_commits} commits...", total=None)
        candidates = await miner.mine_commits(max_commits=max_commits)
        progress.update(task, description="Done!", completed=True)

    if not candidates:
        console.print("[yellow]No decision candidates found in recent commits.[/yellow]")
        return

    table = Table(title=f"Decision Candidates ({len(candidates)} found)")
    table.add_column("#", style="dim")
    table.add_column("Title")
    table.add_column("Source")
    table.add_column("Confidence", justify="right")
    table.add_column("Keywords")

    for i, c in enumerate(candidates[:20], 1):
        table.add_row(
            str(i),
            c.title[:60],
            f"{c.source_type}:{c.source_ref[:8]}",
            f"{c.confidence:.0%}",
            ", ".join(c.keywords_matched[:3]),
        )
    console.print(table)

    if save:
        store = DecisionStore(config.decisions, project_root)
        saved = 0
        for candidate in candidates:
            decision_id = await store.next_id()
            record = candidate.to_decision_record(decision_id)
            await store.save(record)
            saved += 1
        console.print(f"\n[green]Saved {saved} draft decisions.[/green] Review them in .codebase-intel/decisions/")
    else:
        console.print("\nRun with `--save` to save these as draft decisions.")


# -------------------------------------------------------------------
# detect-patterns
# -------------------------------------------------------------------


@app.command(name="detect-patterns")
def detect_patterns(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    save: bool = typer.Option(False, "--save", "-s", help="Save as draft contract"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Auto-detect code patterns and generate quality contracts."""
    _setup_logging(verbose)
    asyncio.run(_detect_patterns_async(path.resolve(), save))


async def _detect_patterns_async(project_root: Path, save: bool) -> None:
    from codebase_intel.contracts.auto_generator import AutoContractGenerator

    generator = AutoContractGenerator(project_root)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing codebase patterns...", total=None)
        patterns = await generator.analyze()
        progress.update(task, description="Done!", completed=True)

    if not patterns:
        console.print("[yellow]No strong patterns detected (need 70%+ consistency).[/yellow]")
        return

    console.print(Panel(
        f"Found [bold]{len(patterns)}[/bold] consistent patterns in your codebase",
        title="Auto-Detected Patterns",
    ))

    table = Table()
    table.add_column("#", style="dim")
    table.add_column("Pattern", style="bold")
    table.add_column("Confidence", justify="right", style="cyan")
    table.add_column("Follows", justify="right", style="green")
    table.add_column("Violates", justify="right", style="red")
    table.add_column("Type")

    for i, p in enumerate(patterns, 1):
        table.add_row(
            str(i),
            p.name,
            f"{p.confidence:.0%}",
            str(p.occurrences),
            str(p.violations),
            p.kind.value,
        )
    console.print(table)

    if save:
        contract = generator.generate_contract(patterns)
        import yaml
        contracts_dir = project_root / ".codebase-intel" / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        out_path = contracts_dir / "auto-detected.yaml"
        data = contract.model_dump(mode="json")
        out_path.write_text(
            yaml.dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        console.print(f"\n[green]Saved contract to {out_path.relative_to(project_root)}[/green]")
        console.print("Review the rules and adjust severity levels as needed.")
    else:
        console.print("\nRun with `--save` to generate a draft contract from these patterns.")


# -------------------------------------------------------------------
# serve
# -------------------------------------------------------------------


@app.command()
def serve(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the MCP server (stdio transport)."""
    _setup_logging(verbose)
    console.print("[dim]Starting MCP server over stdio...[/dim]", err=True)

    from codebase_intel.mcp.server import run_server
    asyncio.run(run_server(path.resolve()))


# -------------------------------------------------------------------
# intent
# -------------------------------------------------------------------


@app.command()
def intent(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    verify: bool = typer.Option(False, "--verify", "-v", help="Run verification on all active intents"),
) -> None:
    """Show and verify tracked intents — what was requested vs what's delivered."""
    asyncio.run(_intent_async(path.resolve(), verify))


async def _intent_async(project_root: Path, verify: bool) -> None:
    from codebase_intel.intent.store import IntentStore
    from codebase_intel.intent.verifier import IntentVerifier

    intents_dir = project_root / ".codebase-intel" / "intents"
    store = IntentStore(intents_dir)
    all_intents = store.load_all()

    if not all_intents:
        console.print("[yellow]No intents tracked yet.[/yellow]")
        console.print("Use the MCP tool `set_intent` to capture goals with acceptance criteria.")
        return

    if verify:
        verifier = IntentVerifier(project_root)
        console.print()

        for intent_obj in all_intents:
            if intent_obj.status.value not in ("active", "partial"):
                continue

            report = await verifier.verify(intent_obj)

            # Update stored state
            updated_criteria = []
            for r in report.results:
                updated_criteria.append(r.criterion.model_copy(update={
                    "verified": r.passed,
                    "failure_reason": r.failure_reason,
                }))
            store.update_criteria(intent_obj.id, updated_criteria)
            store.update_status(intent_obj.id, report.status)

            # Display
            status_color = {
                "verified": "green",
                "partial": "yellow",
                "failed": "red",
                "active": "cyan",
            }.get(report.status.value, "white")

            console.print(Panel(
                f"[bold]{intent_obj.title}[/bold]\n"
                f"Status: [{status_color}]{report.status.value.upper()}[/{status_color}] — "
                f"{report.passed_count}/{report.total} criteria met",
                title=f"Intent {intent_obj.id}",
            ))

            for r in report.results:
                icon = "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]"
                console.print(f"  {icon} {r.criterion.description}")
                if r.failure_reason:
                    console.print(f"       [dim]{r.failure_reason}[/dim]")
            console.print()

    else:
        # Just show status
        table = Table(title="Tracked Intents")
        table.add_column("ID", style="dim")
        table.add_column("Title", style="bold")
        table.add_column("Status")
        table.add_column("Progress", justify="right")
        table.add_column("Gaps", justify="right", style="red")

        for i in all_intents:
            status_color = {
                "verified": "green",
                "partial": "yellow",
                "failed": "red",
                "active": "cyan",
                "abandoned": "dim",
            }.get(i.status.value, "white")

            table.add_row(
                i.id,
                i.title[:50],
                f"[{status_color}]{i.status.value}[/{status_color}]",
                f"{i.criteria_met}/{i.criteria_total} ({i.completion_pct:.0f}%)",
                str(len(i.gaps)),
            )
        console.print(table)
        console.print("\nRun with `--verify` to check all criteria against the actual codebase.")


# -------------------------------------------------------------------
# status
# -------------------------------------------------------------------


@app.command()
def status(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
) -> None:
    """Show component health status."""
    project_root = path.resolve()
    intel_dir = project_root / ".codebase-intel"

    table = Table(title=f"codebase-intel status: {project_root.name}")
    table.add_column("Component", style="bold")
    table.add_column("Status")
    table.add_column("Details")

    # Check config
    config_file = intel_dir / "config.yaml"
    if config_file.exists():
        table.add_row("Config", "[green]OK[/green]", str(config_file.relative_to(project_root)))
    else:
        table.add_row("Config", "[red]Missing[/red]", "Run `codebase-intel init`")

    # Check graph
    graph_db = intel_dir / "graph.db"
    if graph_db.exists():
        size_mb = graph_db.stat().st_size / (1024 * 1024)
        table.add_row("Code Graph", "[green]OK[/green]", f"{size_mb:.1f} MB")
    else:
        table.add_row("Code Graph", "[red]Not built[/red]", "Run `codebase-intel init`")

    # Check decisions
    decisions_dir = intel_dir / "decisions"
    if decisions_dir.exists():
        count = len(list(decisions_dir.glob("*.yaml")))
        table.add_row("Decisions", "[green]OK[/green]", f"{count} records")
    else:
        table.add_row("Decisions", "[yellow]Empty[/yellow]", "Run `codebase-intel mine`")

    # Check contracts
    contracts_dir = intel_dir / "contracts"
    if contracts_dir.exists():
        count = len(list(contracts_dir.glob("*.yaml")))
        table.add_row("Contracts", "[green]OK[/green]", f"{count} files + builtins")
    else:
        table.add_row("Contracts", "[yellow]Builtins only[/yellow]", "")

    console.print(table)


# -------------------------------------------------------------------
# benchmark
# -------------------------------------------------------------------


@app.command()
def benchmark(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run benchmarks — measure token efficiency on this project."""
    _setup_logging(verbose)
    asyncio.run(_benchmark_async(path.resolve()))


async def _benchmark_async(project_root: Path) -> None:
    from codebase_intel.analytics.benchmark import BenchmarkRunner
    from codebase_intel.analytics.tracker import AnalyticsTracker

    tracker = AnalyticsTracker(project_root / ".codebase-intel" / "analytics.db")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running benchmarks...", total=None)

        runner = BenchmarkRunner(project_root)
        report = await runner.run(tracker=tracker)

        progress.update(task, description="Done!", completed=True)

    # Display results
    console.print()
    console.print(Panel(
        f"[bold]{report.repo_name}[/bold]\n"
        f"{report.total_files} files | {report.total_nodes} nodes | "
        f"{report.total_edges} edges | Built in {report.build_time_ms:.0f}ms",
        title="Benchmark Results",
    ))

    table = Table(title="Token Efficiency — Before vs After")
    table.add_column("Scenario", style="bold")
    table.add_column("Naive\n(without tool)", justify="right", style="red")
    table.add_column("Graph Only", justify="right", style="yellow")
    table.add_column("Full Pipeline", justify="right", style="green")
    table.add_column("Reduction", justify="right", style="bold cyan")
    table.add_column("Multiplier", justify="right", style="bold cyan")
    table.add_column("Decisions", justify="right")
    table.add_column("Contracts", justify="right")

    for s in report.scenarios:
        table.add_row(
            s.name,
            f"{s.naive_tokens:,}",
            f"{s.graph_tokens:,}",
            f"{s.full_tokens:,}",
            f"{s.naive_vs_full_reduction:.0f}%",
            f"{s.multiplier:.1f}x",
            str(s.decisions_surfaced),
            str(s.contracts_applied),
        )

    # Average row
    table.add_row(
        "[bold]Average[/bold]",
        "", "", "",
        f"[bold]{report.avg_reduction_pct:.0f}%[/bold]",
        f"[bold]{report.avg_multiplier:.1f}x[/bold]",
        f"[bold]{report.total_decisions_surfaced}[/bold]",
        f"[bold]{report.total_contracts_applied}[/bold]",
        style="on grey23",
    )
    console.print(table)

    # Before/After summary
    console.print()
    console.print("[bold]What this means:[/bold]")
    console.print(
        f"  Without codebase-intel: agent reads [red]{report.scenarios[0].naive_tokens:,}[/red] tokens "
        f"(every file in the directory)"
    ) if report.scenarios else None
    console.print(
        f"  With codebase-intel:    agent reads [green]{report.scenarios[0].full_tokens:,}[/green] tokens "
        f"(only what matters + decisions + contracts)"
    ) if report.scenarios else None
    if report.total_decisions_surfaced > 0:
        console.print(
            f"  [cyan]{report.total_decisions_surfaced}[/cyan] architectural decisions surfaced "
            f"that the agent would have missed"
        )
    if report.total_contracts_applied > 0:
        console.print(
            f"  [cyan]{report.total_contracts_applied}[/cyan] quality rules enforced "
            f"before code was generated"
        )

    tracker.close()


# -------------------------------------------------------------------
# dashboard
# -------------------------------------------------------------------


@app.command()
def dashboard(
    path: Annotated[Path, typer.Argument(help="Project root")] = Path("."),
) -> None:
    """Show live efficiency dashboard — proves value over time."""
    project_root = path.resolve()
    analytics_db = project_root / ".codebase-intel" / "analytics.db"

    if not analytics_db.exists():
        console.print(
            "[yellow]No analytics data yet.[/yellow] Run `codebase-intel benchmark` "
            "or use the MCP server to start collecting data."
        )
        raise typer.Exit(0)

    from codebase_intel.analytics.tracker import AnalyticsTracker

    tracker = AnalyticsTracker(analytics_db)
    stats = tracker.get_lifetime_stats()
    comparison = tracker.get_before_after_comparison()
    benchmarks = tracker.get_benchmark_results()

    # --- Header ---
    console.print()
    console.print(Panel(
        f"[bold]codebase-intel Dashboard[/bold] — {project_root.name}",
        subtitle="Live efficiency tracking",
    ))

    # --- Before/After ---
    if comparison.get("has_data"):
        console.print()
        before_after = Table(title="Before vs After — Token Efficiency")
        before_after.add_column("Metric", style="bold")
        before_after.add_column("Without\ncodebase-intel", justify="right", style="red")
        before_after.add_column("With\ncodebase-intel", justify="right", style="green")
        before_after.add_column("Improvement", justify="right", style="bold cyan")

        b = comparison["before"]
        a = comparison["after"]
        imp = comparison["improvement"]

        before_after.add_row(
            "Avg tokens / request",
            f"{b['tokens_per_request']:,}",
            f"{a['tokens_per_request']:,}",
            f"{imp['token_reduction_pct']:.0f}% fewer ({imp['multiplier']:.1f}x)",
        )
        before_after.add_row(
            "Decisions available",
            "0 (blind)",
            f"{a['decisions_available']}",
            f"{imp['decisions_that_prevented_mistakes']} surfaced",
        )
        before_after.add_row(
            "Contract enforcement",
            "0 (no guardrails)",
            f"{a['contract_checks']}",
            f"{imp['violations_caught_before_generation']} rules enforced",
        )
        before_after.add_row(
            "Drift awareness",
            "None",
            "Active",
            "Stale context detected",
        )
        before_after.add_row(
            "Knows WHY code exists",
            "[red]No[/red]",
            "[green]Yes[/green]",
            "Decision provenance",
        )
        console.print(before_after)

    # --- Lifetime stats ---
    tokens = stats["tokens"]
    quality = stats["context_quality"]

    console.print()
    lifetime = Table(title="Lifetime Statistics")
    lifetime.add_column("Metric", style="bold")
    lifetime.add_column("Value", justify="right")

    lifetime.add_row("Total context requests", f"{stats['total_requests']:,}")
    lifetime.add_row("Total tokens saved", f"[green]{tokens['total_saved']:,}[/green]")
    lifetime.add_row("Token reduction", f"[cyan]{tokens['reduction_pct']:.0f}%[/cyan]")
    lifetime.add_row("Decisions surfaced", f"{quality['decisions_surfaced']:,}")
    lifetime.add_row("Contracts applied", f"{quality['contracts_applied']:,}")
    lifetime.add_row("Drift warnings", f"{quality['drift_warnings']:,}")
    lifetime.add_row("Avg assembly time", f"{stats['performance']['avg_assembly_ms']:.0f}ms")
    console.print(lifetime)

    # --- Benchmark history ---
    if benchmarks:
        console.print()
        bench_table = Table(title="Benchmark History")
        bench_table.add_column("Date", style="dim")
        bench_table.add_column("Project", style="bold")
        bench_table.add_column("Files", justify="right")
        bench_table.add_column("Nodes", justify="right")
        bench_table.add_column("Reduction", justify="right", style="cyan")
        bench_table.add_column("Build Time", justify="right")

        for b in benchmarks[:10]:
            bench_table.add_row(
                b["timestamp"][:10],
                b["repo_name"],
                str(b["total_files"]),
                str(b["total_nodes"]),
                f"{b['avg_token_reduction_pct']:.0f}%",
                f"{b['build_time_ms']:.0f}ms",
            )
        console.print(bench_table)

    # --- Daily trend (sparkline-style) ---
    daily = tracker.get_daily_trend(14)
    if daily:
        console.print()
        console.print("[bold]Daily Token Savings (last 14 days)[/bold]")
        max_saved = max((d["total_naive_tokens"] - d["total_full_tokens"]) for d in daily) or 1
        for d in daily:
            saved = d["total_naive_tokens"] - d["total_full_tokens"]
            bar_len = int((saved / max_saved) * 40)
            bar = "█" * bar_len
            console.print(
                f"  {d['date']}  [green]{bar}[/green] "
                f"{saved:,} tokens saved ({d['total_requests']} requests)"
            )

    tracker.close()


if __name__ == "__main__":
    app()
