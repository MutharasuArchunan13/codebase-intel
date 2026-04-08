# How codebase-intel compares

An honest breakdown of what exists, what each tool does well, and what codebase-intel adds.

## The landscape (April 2026)

### Code Graph / Context Tools (Open Source)

| Tool | Stars | What it does | What it doesn't do |
|---|---:|---|---|
| **code-review-graph** | 6.3K | Code graph + blast radius for token-efficient reviews | No decisions, no contracts, no drift detection |
| **Aider** | 25K+ | Terminal AI coding with repo-map context | No persistent graph, no decisions, no quality enforcement |
| **Continue.dev** | 20K+ | Open-source AI code assistant with codebase indexing | RAG-based (text chunks, not semantic graph), no decisions |
| **Cody (Sourcegraph)** | 3K+ | Code intelligence with embeddings | No decisions, no contracts, enterprise-focused |

### Paid AI Coding Tools

| Tool | Pricing | Context approach | Gaps |
|---|---|---|---|
| **Cursor** | $20/mo | Codebase indexing + RAG | Black box, no decisions, no exportable context |
| **GitHub Copilot** | $10-39/mo | Neighboring files + RAG | No graph, no decisions, no quality contracts |
| **CodeRabbit** | $12/mo | AI code review | Review only (post-generation), no pre-generation context |
| **Codex/Devin** | Varies | Full repo analysis | Autonomous but no persistent decisions or contracts |

### ADR / Decision Tools

| Tool | What it does | Gaps |
|---|---|---|
| **adr-tools** | CLI for markdown ADR files | Manual only, no code linking, no mining, no AI integration |
| **Log4brains** | ADR management with UI | No code anchors, no machine-queryable format, no MCP |

## What makes codebase-intel unique

No single tool combines all of these:

### 1. Decision Provenance (nobody else does this for AI)

Every tool can tell an agent *what* code exists. No tool tells an agent *why* it exists.

- **Auto-mined from git history** — not manual like adr-tools
- **Code-anchored** — linked to specific file locations, not floating documents
- **Machine-queryable via MCP** — agents can ask "why is auth done this way?"
- **Temporal** — review dates, expiry dates, supersedes chains
- **Conflicting decisions detected** — surfaces contradictions before they cause bugs

### 2. Quality Contracts with AI Anti-Pattern Detection

Linters check syntax. CodeRabbit reviews after the fact. Contracts guide AI *before* it writes code.

- **Pre-generation guidance** — agent knows the rules before writing a single line
- **AI-specific rules** — catches hallucinated imports, over-abstraction, YAGNI violations
- **Auto-generated from patterns** — `detect-patterns` scans your code and creates rules automatically
- **Community packs** — FastAPI, React/TS, Node.js/Express

### 3. Feedback Loop (the moat)

No other tool closes the loop between context provided and output quality:

- Tracks acceptance/rejection of AI-generated code
- Learns which context patterns lead to better output
- Maps rejection reasons to actionable improvements
- Proves value: "Sessions with decisions had 91% acceptance rate vs 64% without"

### 4. Live Analytics & Benchmarks

Every other tool says "trust us, it's better." We say "run `benchmark` and see the numbers":

- Reproducible benchmarks against your actual project
- Before/after token comparison
- Daily efficiency trends
- Dashboard showing improvement over time

### 5. Drift Detection

Context rots. No other tool detects this:

- Decisions that reference deleted code
- Expired constraints
- Graph staleness
- Context rot (>30% of records stale = systemic alert)

## Feature matrix

| Feature | codebase-intel | code-review-graph | Aider | Continue | Cursor | Copilot |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Code graph | Yes (19 langs) | Yes (19 langs) | Repo-map | RAG | RAG | RAG |
| Decision journal | **Yes** | No | No | No | No | No |
| Quality contracts | **Yes** | No | No | No | No | No |
| AI anti-patterns | **Yes** | No | No | No | No | No |
| Auto-contract gen | **Yes** | No | No | No | No | No |
| Feedback loop | **Yes** | No | No | No | No | No |
| Drift detection | **Yes** | No | No | No | No | No |
| Token budgeting | **Yes** | Implicit | No | No | N/A | N/A |
| Benchmarks | **Yes** | Yes | No | No | No | No |
| Live dashboard | **Yes** | No | No | No | No | No |
| MCP server | Yes | Yes | No | No | No | No |
| Open source | Yes | Yes | Yes | Yes | No | No |
| Free | Yes | Yes | Yes | Yes | No | No |

## The honest take

**code-review-graph** is great for reviews. Use it for blast radius analysis.

**Aider** is great for AI-assisted coding in the terminal.

**Cursor/Copilot** are great for real-time autocomplete.

**codebase-intel** fills the gap none of them address: *why does this code exist, what rules must AI follow, and is the context still valid?*

You can use codebase-intel alongside any of these tools. They're complementary, not competitive.
