"""Run the Duplicate Finder over a pinned dup sample and save each result.

    python -m evals.run_dup_finder --split dev --run-id dupdev-20b-v1
    python -m evals.score_dup_run dupdev-20b-v1

Resumable like run_labeler: re-run the same command to continue after a stop.
"""
import argparse
import time

from ingestion.db import get_db_connection          # also loads .env

from groq import RateLimitError
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from evals.dataset import REPO
from evals.dup_dataset import load_dup_set
from triage.duplicate_finder import MODEL, make_finder
from triage.tools import SEARCH_TOOL

SAVE = """
INSERT INTO dup_eval_results (run_id, repo, issue_number, split, kind, model, returned,
    invented, rounds, queries, reads, prompt_tokens, completion_tokens, error)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, repo, issue_number) DO UPDATE SET
    returned = EXCLUDED.returned, invented = EXCLUDED.invented, rounds = EXCLUDED.rounds,
    queries = EXCLUDED.queries, reads = EXCLUDED.reads,
    prompt_tokens = EXCLUDED.prompt_tokens, completion_tokens = EXCLUDED.completion_tokens,
    error = EXCLUDED.error, created_at = now()
"""


def tool_use(messages):
    """The agent's own search queries (the seed call has id 'seed') and the issues it read."""
    queries, reads = [], []
    for m in messages:
        for call in getattr(m, "tool_calls", None) or []:
            if call["name"] == SEARCH_TOOL and call["id"] != "seed":
                queries.append(call["args"].get("query", ""))
            elif call["name"] == "read_issue":
                reads.append(call["args"].get("number"))
    return queries, reads


def main(index_factory=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int, default=None)
    # One issue is ~3-5K tokens over 2-3 calls; the free tier allows 8K per minute.
    p.add_argument("--sleep", type=float, default=25.0)
    args = p.parse_args()

    with get_db_connection() as conn:
        if index_factory is None:
            from retrieval.vector import VectorIndex      # loads the embedding model
            index = VectorIndex(conn, REPO)
        else:
            index = index_factory(conn)
        finder = make_finder(conn, index)
        handler = CallbackHandler()

        cases = load_dup_set(conn, args.split)
        if args.limit:
            cases = cases[: args.limit]
        done = {r[0] for r in conn.execute(
            "SELECT issue_number FROM dup_eval_results WHERE run_id = %s AND error IS NULL",
            (args.run_id,),
        ).fetchall()}

        for i, (num, title, body, created_at, kind, originals) in enumerate(cases, 1):
            if num in done:
                continue
            try:
                r = finder.invoke(
                    {"number": num, "title": title, "body": body, "created_at": created_at},
                    config={"callbacks": [handler], "run_name": f"{args.run_id} #{num}",
                            "metadata": {"langfuse_session_id": args.run_id}},
                )
                queries, reads = tool_use(r["messages"])
                row = (r["duplicates"], r["invented"], r["rounds"], queries, reads,
                       r["prompt_tokens"], r["completion_tokens"], None)
                hit = "" if kind == "control" else ("hit " if set(r["duplicates"]) & originals else "miss")
                print(f"[{i}/{len(cases)}] {kind:9} {hit:4} #{num} -> {r['duplicates']} "
                      f"rounds={r['rounds']} searches={len(queries)} reads={len(reads)} "
                      f"tokens={r['prompt_tokens']}+{r['completion_tokens']}")
            except RateLimitError as e:
                print(f"Rate limited at issue {i}. Re-run the same command later to resume.\n{e}")
                break
            except Exception as e:
                row = (None, None, None, None, None, None, None, repr(e))
                print(f"[{i}/{len(cases)}] ERROR #{num}: {e!r}")

            conn.execute(SAVE, (args.run_id, REPO, num, args.split, kind, MODEL, *row))
            conn.commit()
            time.sleep(args.sleep)

    get_client().flush()


if __name__ == "__main__":
    main()
