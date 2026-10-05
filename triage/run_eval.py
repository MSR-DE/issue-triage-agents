"""Run the triage graph over a pinned eval split and save to eval_results.

    python -m triage.run_eval --split dev --run-id dev-20b-graph-v2
    python -m evals.score_run dev-20b-graph-v2

Same table, row format and resume rule as evals/run_labeler.py, so the graph's
runs are scored and compared exactly like the direct-call runs.
"""
import argparse
import time

from ingestion.db import get_db_connection          # also loads .env

from groq import RateLimitError
from langfuse import get_client
from langfuse.langchain import CallbackHandler

from evals.dataset import REPO, load_dev_set, load_test_set
from evals.run_labeler import SAVE                    # one INSERT for both runners
from triage.graph import MODEL, build_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=6.0)   # pacing for 8K tokens/min
    args = p.parse_args()

    graph = build_graph()
    handler = CallbackHandler()

    with get_db_connection() as conn:
        issues = load_dev_set(conn) if args.split == "dev" else load_test_set(conn)
        if args.limit:
            issues = issues[: args.limit]

        # Resume: skip issues this run already labeled successfully.
        done = {r[0] for r in conn.execute(
            "SELECT issue_number FROM eval_results WHERE run_id = %s AND predicted IS NOT NULL",
            (args.run_id,),
        ).fetchall()}

        for i, (num, title, body, gold) in enumerate(issues, 1):
            if num in done:
                continue
            try:
                result = graph.invoke(
                    {"number": num, "title": title, "body": body},
                    config={"callbacks": [handler], "run_name": f"{args.run_id} #{num}"},
                )
                row = (result["label"], result["raw_output"],
                       result["prompt_tokens"], result["completion_tokens"], None)
                mark = "ok " if result["label"] == gold else "XX "
                print(f"[{i}/{len(issues)}] {mark}#{num} gold={gold} pred={result['label']} "
                      f"tokens={result['prompt_tokens']}+{result['completion_tokens']}")
            except RateLimitError as e:
                # Raised only after ChatGroq's own retries (max_retries=2) give up.
                print(f"Rate limited at issue {i}. Re-run the same command later to resume.\n{e}")
                break
            except Exception as e:
                row = (None, None, None, None, repr(e))
                print(f"[{i}/{len(issues)}] ERROR #{num}: {e!r}")

            conn.execute(SAVE, (args.run_id, REPO, num, args.split, MODEL, gold, *row))
            conn.commit()                      # saved before the next call
            time.sleep(args.sleep)

    get_client().flush()


if __name__ == "__main__":
    main()