# r/ClaudeAI Post

**Subreddit:** r/ClaudeAI

**Title:**
I built an MCP server that gives Claude Code something no other tool does — it tells Claude WHY your code exists

**Body:**

Been using Claude Code daily across 11 production services. Claude is incredible at writing code, but it keeps:

- Proposing approaches my team already rejected
- Missing compliance constraints
- Not knowing what else breaks when it changes a file
- Over-engineering things (base classes with one subclass, error handling for impossible conditions)

So I built **codebase-intel** — an MCP server that gives Claude structured context before it writes code.

## What Claude gets from codebase-intel

When you ask Claude to work on a file, the MCP server provides:

1. **Relevant files** — not the whole project, just what's connected (graph-based, 48-75% fewer tokens)
2. **Decisions** — "Your team chose token bucket over sliding window. Here's why. SLA: <2ms p99."
3. **Quality rules** — "This project uses async for all handlers. Don't use sync. All DB access goes through repositories."
4. **Drift warnings** — "Decision DEC-012 references a deleted file. Verify before following."

## MCP setup (30 seconds)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "codebase-intel": {
      "command": "codebase-intel",
      "args": ["serve", "/path/to/your/project"]
    }
  }
}
```

## Unique feature: auto-detects your patterns

```bash
codebase-intel detect-patterns --save
```

Scanned a FastAPI backend and found:
- 100% async handlers
- 96% layer separation (routes don't touch DB directly)
- Service class naming convention (XxxService)
- Custom exception handling everywhere

Generated contract rules from these automatically. Now Claude follows them.

## Install

```bash
pip install codebase-intel
codebase-intel init
```

**GitHub:** https://github.com/MutharasuArchunan13/codebase-intel

7 MCP tools: `get_context`, `query_graph`, `get_decisions`, `get_contracts`, `check_drift`, `impact_analysis`, `get_status`

What other MCP tools would be useful for your Claude Code workflow?
