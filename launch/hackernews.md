# Hacker News — Show HN

**Post URL:** https://news.ycombinator.com/submitlink

**Title (max 80 chars):**
Show HN: codebase-intel – Your AI agent doesn't know why your code exists

**URL:**
https://github.com/MutharasuArchunan13/codebase-intel

**Text (paste this in the comment immediately after posting):**

Hi HN,

I've been using AI coding agents (Claude Code, Copilot, Cursor) daily across 11 production services. They're fast — but they keep making the same mistakes:

- Proposing solutions my team already evaluated and rejected 6 months ago
- Ignoring compliance constraints that shaped our auth system
- Writing patterns that don't match our project's conventions
- Not knowing that changing a config file will break billing AND analytics

The root cause: agents know WHAT code exists, but not WHY it exists.

I built codebase-intel to fix this. It's an open-source context layer that sits between your codebase and any AI agent (via MCP), providing:

1. **Decision Journal** — Records of why decisions were made, auto-mined from git history, linked to specific code locations. When an agent touches rate limiting, it sees: "We chose token bucket over sliding window because memory overhead at scale. SLA: <2ms p99."

2. **Quality Contracts** — Executable rules AI must follow. Not just linting — project-specific architectural patterns. Includes AI anti-pattern detection (hallucinated imports, over-abstraction, YAGNI).

3. **Auto-Contract Generation** — Scans your codebase and detects patterns automatically. On a FastAPI backend it found: 100% async handlers, 96% layer separation, service naming conventions — and generated rules from them.

4. **Feedback Loop** — Tracks whether AI output was accepted or rejected, maps rejection reasons to improvements. No other tool closes this loop.

Benchmarked on 4 production projects: 48-75% token reduction + decisions surfaced + contracts enforced.

Technical: Python, tree-sitter (19 languages), SQLite, MCP server, 595 tests.

Install: `pip install codebase-intel`

Would love feedback on the approach. The comparison with existing tools is at docs/COMPARISON.md — tried to be honest about what others do well and where we fit.
