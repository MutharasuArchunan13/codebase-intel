# Contributing to codebase-intel

Thanks for wanting to help. Here's how to get started.

## Setup

```bash
git clone https://github.com/MutharasuArchunan13/codebase-intel.git
cd codebase-intel
uv sync --dev          # or: pip install -e ".[dev]"
```

## Development workflow

```bash
# Run tests
uv run pytest tests/unit/ -x -q

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check
uv run mypy src/ --ignore-missing-imports

# Run against a real project
uv run codebase-intel init /path/to/your/project
uv run codebase-intel benchmark /path/to/your/project
```

## What to work on

### High-impact areas

1. **Contract packs** — Write quality rules for your framework (Django, Spring Boot, Next.js, etc.). Drop a YAML file in `community-contracts/`.

2. **Language extraction** — Improve parsing for specific languages in `src/codebase_intel/graph/parser.py`. The generic extractor works but language-specific extractors (like Python and JS already have) produce better results.

3. **Decision mining** — Improve git history analysis in `src/codebase_intel/decisions/miner.py`. Better keyword detection, PR description parsing, code review comment extraction.

4. **Benchmarks** — Run `codebase-intel benchmark` on your repos and share results. Real numbers from diverse projects strengthen the case.

5. **Auto-pattern detection** — Add new pattern detectors in `src/codebase_intel/contracts/auto_generator.py`. The more conventions we detect automatically, the lower the adoption barrier.

### Architecture rules

- **Pydantic v2 models** for all structured data — never raw dicts
- **Async-first** for all I/O operations
- **Repository pattern** for storage — models never touch SQLite directly
- **Type hints** on every function (params + return)
- **Custom exceptions** with structured context — never bare try/except

### Code quality

- Run `ruff check` and `ruff format` before committing
- Tests should cover edge cases, not just happy paths
- Comments explain *why*, not *what*

## Pull requests

- Keep PRs focused — one feature or fix per PR
- Include test coverage for new code
- Update `CLAUDE.md` if you change architecture

## Community contract packs

To contribute a contract pack:

1. Create a YAML file in `community-contracts/`
2. Include 5+ rules with clear descriptions
3. Add `fix_suggestion` for every rule
4. Test it: copy to a project's `.codebase-intel/contracts/` and run `codebase-intel benchmark`

See existing packs (`fastapi.yaml`, `react-typescript.yaml`, `nodejs-express.yaml`) for the format.
