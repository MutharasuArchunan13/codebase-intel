# Codebase Intel — Project CLAUDE.md

## Project Overview
Codebase Intelligence Platform: an open-source, agent-agnostic system that provides AI coding agents with structured context, decision provenance, and quality contracts. Solves the three biggest gaps in AI-assisted development: context/memory, judgment/business context, and quality assurance.

## Tech Stack
- **Language:** Python 3.11+
- **Type System:** Pydantic v2 for all models, mypy strict mode
- **Storage:** SQLite via aiosqlite (zero-dependency, portable)
- **Parsing:** tree-sitter for language-agnostic AST analysis
- **Agent Interface:** MCP (Model Context Protocol) server
- **CLI:** Typer + Rich
- **Git Integration:** GitPython
- **Token Counting:** tiktoken
- **Hashing:** xxhash for content fingerprinting

## Architecture
Layered system with five core modules + two interface layers:

```
AI Agent (any) → MCP Server / CLI
                      ↓
            Context Orchestrator
          ↙         ↓          ↘
   Code Graph   Decisions   Contracts
          ↘         ↓          ↙
            Drift Detector
                  ↓
             Codebase (git)
```

### Module Responsibilities:
- **core/** — Shared types, config, exceptions. No business logic.
- **graph/** — Semantic code graph: AST parsing, dependency mapping, impact analysis, SQLite storage
- **decisions/** — Decision journal: structured records, git mining, code linking, temporal validation
- **contracts/** — Quality contracts: architectural rules, pattern libraries, evaluation engine
- **orchestrator/** — Context assembly: budget management, freshness scoring, conflict detection
- **drift/** — Drift detection: staleness, pattern violations, knowledge decay
- **mcp/** — MCP server: exposes all modules as queryable tools for AI agents
- **cli/** — CLI interface: init, analyze, query, serve commands

## Project Flow
1. `codebase-intel init` → scans repo, builds initial code graph, generates starter configs
2. Git hooks keep graph updated incrementally on each commit
3. AI agent connects via MCP → sends task description
4. Orchestrator assembles relevant context (files, decisions, contracts) within token budget
5. Agent receives structured context, writes code
6. Drift detector flags violations post-commit

## Directory Structure
```
src/codebase_intel/
├── core/          # types.py, config.py, exceptions.py
├── graph/         # models.py, storage.py, parser.py, query.py, builder.py
├── decisions/     # models.py, store.py, linker.py, miner.py, validator.py
├── contracts/     # models.py, parser.py, evaluator.py, registry.py
├── orchestrator/  # assembler.py, budget.py, scorer.py, conflict.py
├── drift/         # detector.py, reporter.py
├── workspace/     # manager.py, registry.py — global multi-project workspace
├── crossrepo/     # models.py, scanner.py, registry.py — cross-service deps
├── analytics/     # tracker.py, benchmark.py, feedback.py
├── intent/        # models.py, verifier.py, store.py
├── mcp/           # server.py (supports --auto multi-project mode)
└── cli/           # main.py
```

## Key Commands
- `ruff check src/ tests/` — Lint
- `ruff format src/ tests/` — Format
- `mypy src/` — Type check
- `pytest` — Run tests with coverage
- `pytest tests/unit/` — Unit tests only
- `pytest tests/integration/` — Integration tests only

## Conventions
- All models inherit from Pydantic BaseModel — never use raw dicts for structured data
- Async-first: all I/O operations are async
- Repository pattern for storage: models never touch SQLite directly
- Type hints on every function (params + return)
- Custom exceptions over generic ones — always include context
- Content hashing via xxhash for change detection
- ISO 8601 for all timestamps, UTC timezone

## Current Focus
- Initial architecture build-out with comprehensive edge case handling
- Core module implementation
- MCP server interface design
