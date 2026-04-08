# X/Twitter Thread

**Thread (post each as a separate tweet):**

---

**Tweet 1 (hook):**

Your AI coding agent can write code in 30 seconds.

But it doesn't know:
- Why your team chose that architecture
- What compliance constraint shapes your auth
- That changing this file breaks billing

I built an open-source fix. Thread:

---

**Tweet 2 (problem):**

The real problem isn't speed. AI agents are fast.

The problem is CONTEXT:

- "Context rot" — agents forget decisions mid-session
- "Hallucination" — agents reference APIs that don't exist
- "Pattern blindness" — agents ignore your project's conventions

Token windows don't solve this. You need structure.

---

**Tweet 3 (solution):**

codebase-intel gives AI agents 3 things nobody else provides:

1/ Decision Journal — WHY code exists (auto-mined from git)
2/ Quality Contracts — rules AI must follow (auto-detected from your code)
3/ Drift Detection — catches when context goes stale

All via MCP. Works with Claude Code, Cursor, any MCP tool.

---

**Tweet 4 (proof):**

Benchmarked on 4 production codebases:

FastAPI monolith: 63% token reduction + 13 decisions surfaced
Microservice: 75% reduction + 4 contracts enforced

The decisions part is what matters most. Your AI stopped proposing approaches your team already rejected.

---

**Tweet 5 (unique):**

The killer feature: auto-contract generation.

Run one command:
`codebase-intel detect-patterns --save`

It scans your code and finds:
- "100% of handlers use async"
- "96% layer separation"
- "Service classes follow XxxService naming"

Generates rules automatically. Zero manual setup.

---

**Tweet 6 (CTA):**

Open source. MIT licensed. 19 languages.

pip install codebase-intel

GitHub: github.com/MutharasuArchunan13/codebase-intel
PyPI: pypi.org/project/codebase-intel

If your AI agent doesn't know why your code exists, it's guessing. Stop guessing.

---
