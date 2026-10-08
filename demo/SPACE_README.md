---
title: uv issue triage
emoji: 🔧
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
short_description: Multi-agent GitHub issue triage for astral-sh/uv
---

# uv issue triage, live

Pick a real issue from [astral-sh/uv](https://github.com/astral-sh/uv) (or paste your own)
and three agents triage it in parallel: a labeler, a Duplicate Finder and an Investigator
that checks whether a merged PR already fixed it. A drafter writes a first reply, code
checks it, and you approve, edit or reject. Nothing is posted to GitHub.

Built with LangGraph (parallel branches, human-in-the-loop pause saved in Postgres),
Groq (gpt-oss-20b / 120b), pgvector and bge-small embeddings.

Code, evals and write-up: https://github.com/MSR-DE/issue-triage-agents

This Space is deployed automatically by the repo's CI (`deploy/push_space.py`); edit the
code there, not here.
