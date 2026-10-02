import argparse
import time

from ingestion.db import get_db_connection          # also loads .env (GROQ_API_KEY)
from groq import Groq, RateLimitError

from evals.dataset import REPO, load_dev_set, load_test_set
from evals.labeler import label_issue

SAVE = """
INSERT INTO eval_results (run_id, repo, issue_number, split, model, gold,
                          predicted, raw_output, prompt_tokens, completion_tokens, error)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (run_id, repo, issue_number) DO UPDATE SET
    predicted         = EXCLUDED.predicted,
    raw_output        = EXCLUDED.raw_output,
    prompt_tokens     = EXCLUDED.prompt_tokens,
    completion_tokens = EXCLUDED.completion_tokens,
    error             = EXCLUDED.error,
    created_at        = now()
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test"], required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sleep", type=float, default=6.0)   # pacing for 8K tokens/min
    args = p.parse_args()

    client = Groq()
    with get_db_connection() as conn:
        issues = load_dev_set(conn) if args.split == "dev" else load_test_set(conn)
        if args.limit:
            issues = issues[: args.limit]

        # resume: skip issues this run already labeled successfully
        done = {r[0] for r in conn.execute(
            "SELECT issue_number FROM eval_results WHERE run_id = %s AND predicted IS NOT NULL",
            (args.run_id,),
        ).fetchall()}

        for i, (num, title, body, gold) in enumerate(issues, 1):
            if num in done:
                continue
            try:
                label, raw, usage = label_issue(client, args.model, title, body)
                row = (label, raw, usage.prompt_tokens, usage.completion_tokens, None)
                mark = "ok " if label == gold else "XX "
                print(f"[{i}/{len(issues)}] {mark}#{num} gold={gold} pred={label} "
                      f"tokens={usage.prompt_tokens}+{usage.completion_tokens}")
            except RateLimitError as e:
                print(f"Rate limited at issue {i}. Re-run the same command later to resume.\n{e}")
                break
            except Exception as e:
                row = (None, None, None, None, repr(e))
                print(f"[{i}/{len(issues)}] ERROR #{num}: {e!r}")

            conn.execute(SAVE, (args.run_id, REPO, num, args.split, args.model, gold, *row))
            conn.commit()                      # saved before the next call
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()