"""Run the Investigator over a pinned fix sample and save each result.

    python -m evals.run_investigator --split tune --run-id invtune-20b-v1   # look at these by hand
    python -m evals.run_investigator --split eval --run-id inveval-20b-v1   # once, prompts frozen
    python -m evals.score_inv_run inveval-20b-v1

Resumable: re-run the same command to continue after a stop (rate limit, Ctrl+C).
"""
import argparse
import time

from ingestion.db import get_db_connection          # also loads .env

from groq import RateLimitError
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from evals.dataset import REPO
from evals.fix_dataset import load_fix_set
from triage.fix_tools import READ_TOOL, SEARCH_TOOL
from triage.investigator import MODEL, make_index, make_investigator

SAVE = """
INSERT INTO inv_eval_results (run_id, repo, issue_number, split, kind, model, returned,
    invented, reported_version, rounds, queries, reads, prompt_tokens, completion_tokens, error)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, repo, issue_number) DO UPDATE SET
    returned = EXCLUDED.returned, invented = EXCLUDED.invented,
    reported_version = EXCLUDED.reported_version, rounds = EXCLUDED.rounds,
    queries = EXCLUDED.queries, reads = EXCLUDED.reads,
    prompt_tokens = EXCLUDED.prompt_tokens, completion_tokens = EXCLUDED.completion_tokens,
    error = EXCLUDED.error, created_at = now()
"""


def tool_use(messages):
    queries, reads = [], []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] == SEARCH_TOOL and call["id"] != "seed":
                queries.append(call["args"].get("query", ""))
            elif call["name"] == READ_TOOL:
                reads.append(call["args"].get("number"))
    return queries, reads


def main(index_factory=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["tune", "eval"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=25.0)     # 8K tokens/minute on the free tier
    args = p.parse_args()

    with get_db_connection() as conn:
        index = (index_factory or make_index)(conn)
        investigator = make_investigator(conn, index)
        handler = CallbackHandler()
        cases = load_fix_set(conn, args.split)[: args.limit]
        done = {r[0] for r in conn.execute(
            "SELECT issue_number FROM inv_eval_results WHERE run_id = %s AND error IS NULL",
            (args.run_id,)).fetchall()}

        for i, (num, title, body, created_at, kind, fix_prs) in enumerate(cases, 1):
            if num in done:
                continue
            try:
                r = investigator.invoke(
                    {"number": num, "title": title, "body": body, "created_at": created_at},
                    config={"callbacks": [handler], "run_name": f"{args.run_id} #{num}",
                            "metadata": {"langfuse_session_id": args.run_id}},
                )
                queries, reads = tool_use(r["messages"])
                row = (r["fixed_by"], r["invented"], r.get("reported_version"), r["rounds"],
                       queries, reads, r["prompt_tokens"], r["completion_tokens"], None)
                hit = ("" if kind == "control" else
                       "hit " if set(r["fixed_by"]) & fix_prs else "miss")
                print(f"[{i}/{len(cases)}] {kind:7} {hit:4} #{num} -> {r['fixed_by']} "
                      f"v={r.get('reported_version')} rounds={r['rounds']} searches={len(queries)} "
                      f"reads={len(reads)} tokens={r['prompt_tokens']}+{r['completion_tokens']}")
            except RateLimitError as e:
                print(f"Rate limited at issue {i}. Re-run the same command later to resume.\n{e}")
                break
            except Exception as e:
                row = (None, None, None, None, None, None, None, None, repr(e))
                print(f"[{i}/{len(cases)}] ERROR #{num}: {e!r}")

            conn.execute(SAVE, (args.run_id, REPO, num, args.split, kind, MODEL, *row))
            conn.commit()
            time.sleep(args.sleep)

    get_client().flush()


if __name__ == "__main__":
    main()
