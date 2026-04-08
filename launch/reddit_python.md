# r/Python Post

**Subreddit:** r/Python
**Flair:** Showcase
**Post type:** Text post

**Title:**
codebase-intel — gives AI coding agents decision provenance, quality contracts, and drift detection

**Body:**

## What My Project Does

codebase-intel is an open-source Python tool that provides AI coding agents with structured context via MCP (Model Context Protocol). Instead of just telling an agent what code exists, it tells the agent:

- **Why code exists** — Decision records auto-mined from git history, linked to specific file locations. Example: "We chose token bucket over sliding window because of memory overhead at scale. SLA constraint: <2ms p99 latency."
- **What rules to follow** — Quality contracts that enforce project-specific patterns before code is generated. Includes AI-specific anti-pattern detection (hallucinated imports, over-abstraction, YAGNI violations).
- **What's stale** — Drift detection that catches when decisions reference deleted files, expired constraints, or outdated code anchors.

It also auto-detects patterns in your codebase (`codebase-intel detect-patterns`) and generates quality rules from them — zero manual setup. On a 359-file FastAPI project, it detected 100% async handler usage, 96% layer separation, and service naming conventions automatically.

Built with: Python 3.11+, Pydantic v2, tree-sitter-language-pack (19 languages), SQLite, MCP server, tiktoken. 595 tests.

Install: `pip install codebase-intel`

GitHub: https://github.com/MutharasuArchunan13/codebase-intel

## Target Audience

Production use. Built for developers who use AI coding agents (Claude Code, Cursor, Copilot) on real projects with established patterns, architectural decisions, and compliance requirements. Most useful for teams working on microservices, FastAPI backends, or any codebase where "why" matters as much as "what."

Not a toy project — benchmarked on 4 production codebases with 48-75% token reduction and architectural decisions surfaced that agents would have missed.

## Comparison

**vs code-review-graph (6.3K stars):** code-review-graph builds an excellent code graph for blast-radius analysis during reviews. codebase-intel also has a code graph (19 languages) but adds three layers code-review-graph doesn't have: decision journal (why code exists), quality contracts (rules AI must follow), and drift detection (is context still valid). They're complementary — you can use both.

**vs Aider:** Aider uses a repo-map for context. No persistent decisions, no quality enforcement, no drift detection. Different focus — Aider is a coding assistant, codebase-intel is a context layer for any assistant.

**vs Cursor/Copilot:** These use RAG-based indexing (text chunk retrieval). No decision provenance, no architectural contracts, no auto-pattern detection. Closed ecosystems. codebase-intel is open-source and agent-agnostic via MCP.

**vs adr-tools/Log4brains:** These manage Architecture Decision Records manually. codebase-intel auto-mines decisions from git, links them to code locations, makes them queryable by AI agents via MCP, and detects when they go stale. Different generation of tooling.

**Unique features no alternative has:**
- Auto-contract generation from detected codebase patterns
- Feedback loop tracking AI output acceptance/rejection
- AI-specific anti-pattern detection (hallucinated imports, over-abstraction)
- Live efficiency dashboard with reproducible benchmarks
