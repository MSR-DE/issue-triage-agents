# Multi-agent issue triage for astral-sh/uv

![CI](https://github.com/MSR-DE/multi-agent-issue-triage/actions/workflows/ci.yml/badge.svg)

A new GitHub issue goes through three steps in parallel: a **labeler**, a **Duplicate Finder**
agent that searches earlier issues, and an **Investigator** agent that checks whether a merged
PR already fixed it. A drafter then writes a first reply, code checks it, and a maintainer
approves, edits or rejects it before anything is posted.

Built with LangGraph and measured on real issues from [uv](https://github.com/astral-sh/uv), the Python package
manager: about 9,300 issues, 12,600 pull requests and 320 releases.
Where a simpler method exists, each result is compared with it on the same issues.

## Why I built this

uv gets a lot of issues: about 9,300 since late 2023. Each one needs someone to label it, check
whether it was reported before, and check whether a recent change already fixed it. I wanted to
see how much of that work agents can take on, and to measure it honestly instead of building a
demo that only looks good: every agent is compared with a simple baseline on the same real issues,
and a human approves anything before it is posted. Issue text is written by strangers, so I also
attacked my own system with prompt injection and measured what got through.

## Highlights

- **Three agents, one human gate.** The labeler, Duplicate Finder and Investigator run in parallel
  (LangGraph); the run then pauses for a person to approve, edit or reject, and the pause is saved
  in Postgres.
- **Measured, not just demoed.** Each part is compared with a simpler method on real uv issues,
  with confidence intervals. For example, the labeler beats rules that read the issue form, 79.8% vs
  74.6% on 228 test issues, and the Investigator never claimed a fix on 30 bugs that weren't fixed.
- **Attacked on purpose.** 20 hand-written prompt-injection issues aimed at the reply and the label,
  run with and without the protections ([details](#prompt-injection)).
- **No made-up references.** The agents can only cite issues and PRs their own searches returned;
  any other number is dropped in code.
- **Tested in CI** on every push with a scripted fake LLM: no network, no tokens.

## What it looks like

![A triaged issue: proposed label, possible duplicates, possible fix and the drafted reply](docs/triage.png)

![Issue #5210: the Investigator names PR #5148, merged the day before the issue was opened; the Duplicate Finder adds #4988](docs/already-fixed.png)

![The review step: a reviewer's edited reply is checked again before anything would be posted; here it flags a link and a request for a token](docs/review.png)

## Results

| Part | Measured on | Result | Baseline on the same issues |
|---|---|---|---|
| **Labeler** (bug / enhancement / question / documentation) | 228 test issues, run once | **79.8%** correct (95% CI 74–85%). The injection-hardened version scores the same on the dev set (84%, 42/50) | 74.6% for rules that read the issue form; 45.6% for always "bug". Fixed 13 rule errors, broke 1 (p ≈ 0.002) |
| **Duplicate search** (embeddings) | 347 real duplicates | **55.0%** have the original in the top 5 (CI 50–60%) | 32.6% for keyword search (BM25). BM25 + embeddings (hybrid) scored 48.7%, *worse* than embeddings alone (p = 0.014) |
| **Duplicate Finder** agent | 52 duplicates + 15 non-duplicates (test) | Suggests **1.3** issues on average instead of 5, and finds as many originals: 55.8% vs 55.8%. With the same number of suggestions it finds 29/52 vs 20/52 (fixed 9, broke 0, p = 0.004). **57%** of its suggestions are right (search: 17%). On the 15 "non-duplicates" it suggested something for 13; read by hand, 5 of those were real duplicates nobody had marked. On the 10 true non-duplicates it said "none" only twice (4 suggestions borderline, 4 wrong) | plain embedding search, top 5 or the same number of results |
| **Investigator** ("already fixed?") | 24 already-fixed + 30 not-fixed bugs | Never claimed a fix on the 30 unfixed bugs (CI 89–100%). Named the right PR for 11 of 24. Right call 76% (CI 63–85%) | always "not fixed": 56%. Embedding search finds the PR in its top 5 for 16/24, but only 14% of its suggestions are right, vs 69% for the agent |
| **Prompt injection** | 20 hand-written attack issues (18 aim at the reply, 2 at the label) | Reply attacks reaching the reviewer unflagged: **0/18** on the first run, **1/18** on a re-run (the drafter asked the reporter for their password or token; a check now flags this). Label attacks: **0/2** with the hardened labeler (2/2 before) | 5/18 reply attacks got through without the protections, in both runs |
| **Made-up issue / PR numbers** | Duplicate Finder dev and test runs, Investigator eval | 0 invented. Any number an agent's own searches didn't return is dropped in code before a reviewer sees it | |

How to read these:
- "Top 5" counts a duplicate as found when the original issue is among the first five results.
  uv's maintainers often close duplicates into one tracking issue, so finding any issue from the
  same group counts. Strict scoring (exact original only): embeddings 41.5% vs BM25 24.5%.
  Both are in [results/retrieval_v2.txt](results/retrieval_v2.txt).
- The Duplicate Finder usually returns 1–2 issues and search always returns 5, so it is also compared with
  search's top 1–2 ("the same number of suggestions"), where the fair comparison is.
- p-values are exact McNemar tests on the issues where the two methods disagree; CIs are 95%
  Wilson intervals. The agent evals are small (tens of issues), so their intervals are wide.

## Prompt injection

Issue text is written by strangers, so the system treats it as hostile. The protections, in layers:

1. **Marked as untrusted.** Every model gets the issue inside `<issue>` tags with a note to treat it
   as data, never as instructions. The labeler and drafter also neutralise `<issue>` tags inside the
   text, so an issue can't close the block early.
2. **Narrow outputs.** The labeler can only answer one of four labels. The agents have read-only
   tools and can only return issue and PR numbers their own searches found.
3. **Checks in code** on every reply: links, @mentions, code or commands, promises, passwords or
   tokens, and issue numbers the agents didn't find. They run again after a human edits the reply.
4. **A human approves everything.** The GitHub token is read-only, and posting is a dry run.

The eval ([evals/injection_attacks.py](evals/injection_attacks.py)) has 20 hand-written attack
issues: instruction overrides, a fake maintainer voice, hidden HTML comments, a `</issue>` break-out,
base64-encoded instructions, a request for the reporter's token, and two that try to force a label.
Each goes through the labeler, then through the drafter twice: with the protections and without.

| | Without the protections | With the protections |
|---|---|---|
| Reply attacks that reached the reviewer unflagged | 5/18 | **0/18** on the first run; **1/18** on a re-run (it asked for the reporter's token; a new check now flags that) |
| Label attacks that changed the label | 2/2 (original labeler) | **0/2** (hardened labeler, same 84% on the dev set) |

## How it works

```mermaid
flowchart LR
    I[New issue] --> L[Labeler<br/>gpt-oss-20b]
    I --> D[Duplicate Finder<br/>agent, gpt-oss-20b]
    I --> V[Investigator<br/>agent, gpt-oss-20b]
    L --> R[Draft reply<br/>gpt-oss-120b<br/>+ code checks]
    D --> R
    V --> R
    R --> H[Human review<br/>pause saved in Postgres]
    H -->|approve or edit| P[Post<br/>dry run]
    H -->|reject| E[Nothing posted]
```

- **Parallel steps, one human gate.** LangGraph runs the three steps at the same time and waits
  for all of them. The graph then pauses (`interrupt`) and the pause is saved in Postgres, so the
  review can happen in another process or days later.
- **The agents can only cite what they found.** The Duplicate Finder (tools: `search_similar_issues`,
  `read_issue`) and the Investigator (`search_fixes`, `read_pr`) get at most 2 tool rounds, and
  can only read issues and PRs their own searches returned. Any other number in their answer is
  dropped in code and counted.
- **Facts in code, judgement in the model.** The Investigator only searches PRs merged in the 7 days
  before the issue and, when the reporter states their uv version, after that version's release.
  The code works this window out; the model only decides whether a candidate fixes the problem.

## How it was measured

- **Answer keys from what maintainers actually did:** duplicate links and labels for the Duplicate
  Finder; comments like "fixed in #1234" for the Investigator. A hand check of the second key
  found 7 hedged links ("I thought we fixed this…"), so hedged comments are now skipped.
- **Pinned sets, drawn before any results.** The dev sets are for trying things; each test set is
  run once, after the prompt is frozen.
- **Always against a baseline, on the same issues:** rules for the labeler, keyword and embedding
  search for the agents. Reported as "fixed / broke", not just accuracy.
- **Budget:** Groq's free tier allows 200K tokens per model per day, which sets the size of
  every eval. All runners are resumable and log tokens per issue.

## Things that went wrong, and what changed

- **The duplicate answer key was wrong at first.** A trace showed a correct agent answer scored
  as a miss: maintainers close duplicates into tracking issues. That led to the "same group"
  scoring. A pinned "Before posting" guide had linked 38 unrelated issues together; it is excluded.
- **The popular choice lost:** hybrid search was significantly worse than embeddings alone on uv.
- **A word list missed what reading caught.** The check for promises missed "We'll take a look";
  reading 20 drafts found it. The check now flags any "we'll / we will", and the drafter is told
  not to write as the maintainers: replies saying "we'll …" on the injection set went from 16/18
  to 2/18.
- **A second run found a hole the first one missed.** Run again, the same 18 injection attacks got
  one through: the drafter asked the reporter to paste their password or token, and no check
  looked for that. Model output changes from run to run, so one clean run isn't proof. A check for
  passwords, tokens and keys now flags it (it also flags 2 of the other 111 saved replies, both
  about a reporter's own token problem).
- **An LLM judge wasn't reliable enough here.** DeepEval's faithfulness metric (default mode)
  caught 1 of 3 planted bad replies; a stricter mode caught all 3 but also failed correct replies.
  So the gate is code checks plus a human, not a judge.
- **Resuming a saved pause twice replays the first decision** in LangGraph. Found while testing the
  demo: a second "Reject" showed the first "Approve". Each decision now runs on its own fork of the
  pause (`resume_from_pause` in `triage/graph.py`, with tests).
- **Groq rejected the final answer call** when it contained tool-call history ("Tool choice is none").
  The verdict call now gets a plain-text summary of the run and no tools.

## Limitations

- One repository (uv) and small agent eval sets: the intervals above are wide.
- The Investigator's search window was chosen using the same already-fixed issues it is evaluated
  on (there were no others), and the agent is cautious: on 8 of 24 already-fixed issues it said "not fixed",
  in 5 of them after seeing the right PR.
- The Duplicate Finder rarely says "no duplicate": on 10 issues that really have none, it suggested
  something for 8 (4 of them related, 4 wrong). A reviewer dismisses those quickly, but it is noise.
- The "not a duplicate" controls come from issues nobody marked as duplicates; reading them, 5 of 15
  in the test set (and 3 of 10 in the dev set) were real duplicates nobody had marked.
- Nothing is posted to GitHub. Every reply needs a human.

## Repository

```
ingestion/   GitHub API -> Postgres: issues, PRs, releases, comments; the two answer keys
retrieval/   keyword search (BM25), embeddings (bge-small + pgvector), hybrid, PR search
triage/      the LangGraph graph, the agents and their tools, drafter, reply checks
evals/       pinned eval sets, runners, scorers, injection attacks, statistics
tests/       unit tests with a scripted fake LLM: no network, no tokens (run by CI)
demo/        Streamlit app: triage an issue and review it in the browser
deploy/      exports a slim copy of the database to a hosted Postgres (for the demo)
results/     saved eval reports
```

Stack: LangGraph, Groq (gpt-oss-20b and gpt-oss-120b, free tier), Postgres + pgvector,
BAAI/bge-small-en-v1.5 embeddings (local), Langfuse tracing, DeepEval, Streamlit,
GitHub Actions (ruff + pytest on every push).

## Run it

You need Python 3.13, Docker, a Groq API key (free) and a read-only GitHub token.

```bash
# 1. Postgres with pgvector, on port 5433
docker run -d --name gh-pg -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=gh_triage \
  -p 5433:5432 pgvector/pgvector:pg16
psql postgresql://postgres:postgres@localhost:5433/gh_triage -f schema.sql

# 2. Python packages
pip install -r requirements.txt
```

On Windows, `psycopg` uses the system's `libpq`: install the PostgreSQL command-line tools and
put their `bin` folder on `PATH` (or use `psycopg[binary]` if your machine allows it).

Create a `.env` file:

```
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/gh_triage
GITHUB_TOKEN=...            # read-only is enough
GROQ_API_KEY=...
LANGFUSE_PUBLIC_KEY=...     # tracing
LANGFUSE_SECRET_KEY=...
LANGFUSE_BASE_URL=...
```

Load the data (each step can be stopped and re-run; it continues where it stopped):

```bash
python -m ingestion              # issues
python -m ingestion.duplicates   # duplicate answer key
python -m retrieval.embed        # issue embeddings
python -m ingestion.slice4       # releases, PRs, comments, "fixed in" answer key
python -m retrieval.fix_index    # PR embeddings
```

Use it:

```bash
python -m triage.graph run 19540      # label + duplicates + already-fixed, draft, then pause
python -m triage.graph review 19540   # approve / edit / reject -> prints what would be posted
streamlit run demo/app.py             # the same in a browser
pytest -q                             # tests, no network
```

Main evals (each prints the comparison with its baseline):

```bash
python -m triage.run_eval --split test --run-id <id>        && python -m evals.score_run <id>
python -m evals.score_retrieval
python -m evals.run_dup_finder --split test_small --run-id <id> && python -m evals.score_dup_run <id>
python -m evals.run_investigator --split eval --run-id <id> && python -m evals.score_inv_run <id>
python -m evals.run_injection
```
