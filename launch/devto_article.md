---
title: The 3 Things AI Coding Agents Still Can't Do (And How I Fixed Them)
published: true
tags: ai, python, opensource, programming
canonical_url: https://github.com/MutharasuArchunan13/codebase-intel
---

I've been using AI coding agents — Claude Code, GitHub Copilot, Cursor — daily across 11 production services for the past year. They've transformed how I work. But they keep failing at the same three things.

This isn't a rant. This is a problem statement with a solution I built and open-sourced.

## Problem 1: Context Rot

AI agents lose track of what matters during longer sessions. Even with 1M token context windows, having a massive database without proper indexing doesn't help — the data exists, but retrieval is inefficient.

**Real example:** I asked Claude to add rate limiting to a payment endpoint. It read the file, saw no existing rate limiter, and built one from scratch. There was already a rate limiting module in `src/middleware/rate_limiter.py` with 200 lines of battle-tested code. Claude just didn't see it because it was two directories away.

## Problem 2: No Decision Awareness

AI doesn't know WHY your code is the way it is.

**Real example:** Claude suggested switching our auth tokens from EdDSA to RS256 "for broader compatibility." Our team specifically chose EdDSA six months ago because our compliance team required it for a security audit. That decision existed in a PR description that Claude never saw.

The agent was technically correct. It was contextually wrong.

## Problem 3: Quality Drift

AI-generated code prioritizes speed over project conventions.

**Real example:** Our project uses async handlers everywhere — 100% consistency across 46 handler files. Claude generated a sync handler. It worked. Tests passed. But it broke the convention, and the next developer who saw it started writing sync handlers too. One exception became a pattern.

## The Root Cause

Every AI context tool solves **what code exists** — code graphs, RAG, embeddings. Nobody solves **why code exists**, **what rules apply**, and **whether the context is still valid**.

## What I Built

**[codebase-intel](https://github.com/MutharasuArchunan13/codebase-intel)** — an open-source platform that provides AI agents with three things no other tool offers:

### 1. Decision Journal

Structured records of WHY decisions were made, linked to specific code locations:

```yaml
id: DEC-042
title: "Use token bucket for rate limiting"
context: "Payment endpoint hammered during flash sales"
decision: "Token bucket, per-user buckets, 100 req/min"
alternatives:
  - name: sliding_window
    rejection_reason: "Memory overhead too high at scale"
constraints:
  - description: "Must not add >2ms p99 latency"
    source: sla
    is_hard: true
code_anchors:
  - "src/middleware/rate_limiter.py:15-82"
```

These are **auto-mined from git history**. Run `codebase-intel mine --save` and it extracts decision candidates from commit messages and PR descriptions using keyword analysis. On a 359-file FastAPI project, it found 16 decision candidates automatically.

### 2. Quality Contracts

Not linting. Project-specific architectural rules that AI reads BEFORE generating code:

```yaml
rules:
  - id: async-handlers
    name: All handlers must be async
    severity: error
    pattern: "(?<!async\\s)def\\s+(get_|post_|create_)"
    fix_suggestion: "Use async def for handlers"
```

**The killer feature: auto-detection.** Run `codebase-intel detect-patterns --save` and it scans your codebase to find patterns you already follow:

```
┏━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ # ┃ Pattern                ┃ Confidence ┃ Follows ┃ Violates ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 1 │ Async handlers         │       100% │      46 │        0 │
│ 2 │ Layer separation       │        96% │     132 │        5 │
│ 3 │ Service class naming   │        90% │      35 │        0 │
│ 4 │ Custom exceptions      │       100% │      47 │        0 │
│ 5 │ Docstring convention   │        88% │     793 │      108 │
└───┴────────────────────────┴────────────┴─────────┴──────────┘
```

Zero manual contract writing. The tool discovers what your team already does and generates rules from it.

### 3. Drift Detection

Context rots. Decisions go stale. Code anchors point to deleted files. codebase-intel detects this:

```bash
$ codebase-intel drift

Overall: MEDIUM
- Decision DEC-012 anchored to deleted file
- Decision DEC-008 is past its review date
- 2 files changed since last graph index
```

## The Benchmarks

Tested on 4 production codebases:

| Project | Without Tool | With codebase-intel | Reduction | Decisions Surfaced |
|---|---:|---:|---:|---:|
| FastAPI monolith (359 files) | 16,063 tokens | 5,955 tokens | **63%** | 13 |
| Microservice A (358 files) | 14,611 | 5,955 | **59%** | 0 |
| Microservice B (153 files) | 5,904 | 1,476 | **75%** | 0 |

The token reduction is nice. But the real value is the **13 decisions surfaced** — those are constraints, trade-offs, and rejected alternatives that would have been invisible to the agent.

## How It Works

```
AI Agent → MCP Server → Context Orchestrator
                              ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
              Code Graph  Decisions  Contracts
              (19 langs)  (git-mined) (auto-detected)
                    ↓         ↓         ↓
                    └─────────┼─────────┘
                              ↓
                        Drift Detector
```

The agent asks "what do I need to know to work on this file?" and gets:
- Relevant files (from the code graph)
- Applicable decisions (linked to this code area)
- Quality rules to follow
- Warnings about stale context

All fitted within a token budget, prioritized by relevance.

## Getting Started

```bash
pip install codebase-intel
cd your-project
codebase-intel init              # build graph, mine decisions
codebase-intel detect-patterns --save  # auto-generate contracts
codebase-intel benchmark         # see before/after numbers
```

For Claude Code, add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codebase-intel": {
      "command": "codebase-intel",
      "args": ["serve", "/path/to/project"]
    }
  }
}
```

## It's Not a Competition

Tools like code-review-graph (6.3K stars) are excellent for code graphs. Cursor and Copilot are great for autocomplete. We don't replace any of them.

We fill the gap none of them address: **why does this code exist, what rules must AI follow, and is the context still valid?**

## What I'd Love Feedback On

1. What patterns should the auto-detector look for in your stack?
2. What quality rules would you want enforced in your project?
3. Are there decision types we're missing?

**GitHub:** [MutharasuArchunan13/codebase-intel](https://github.com/MutharasuArchunan13/codebase-intel)
**PyPI:** `pip install codebase-intel`
**License:** MIT

---

*Built with Python, tree-sitter, SQLite, and MCP. 10K+ lines of source, 595 tests, 19 languages supported.*
