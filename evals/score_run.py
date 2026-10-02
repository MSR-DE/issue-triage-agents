import sys

from ingestion.db import get_db_connection
from evals.baseline import template_label
from evals.dataset import TYPE_LABELS
from evals.metrics import report


def main():
    run_id = sys.argv[1]
    with get_db_connection() as conn:
        rows = conn.execute("""
            SELECT r.issue_number, r.gold, r.predicted, i.body,
                   r.prompt_tokens, r.completion_tokens
            FROM eval_results r
            JOIN issues i ON i.repo = r.repo AND i.issue_number = r.issue_number
            WHERE r.run_id = %s
            ORDER BY r.issue_number
        """, (run_id,)).fetchall()

    ok = [r for r in rows if r[2] is not None]
    print(f"run {run_id}: {len(ok)} labeled, {len(rows) - len(ok)} failed\n")
    if not ok:
        return

    report([(gold, pred) for _, gold, pred, *_ in ok], TYPE_LABELS)

    # Paired comparison with the template baseline on the same issues
    base_right = sum(template_label(body) == gold for _, gold, _, body, *_ in ok)
    fixed = [num for num, gold, pred, body, *_ in ok
             if pred == gold and template_label(body) != gold]
    broke = [num for num, gold, pred, body, *_ in ok
             if pred != gold and template_label(body) == gold]
    print(f"\ntemplate baseline on these issues: {base_right}/{len(ok)} = {base_right / len(ok):.1%}")
    print(f"model fixed {len(fixed)} baseline errors: {fixed}")
    print(f"model broke {len(broke)} baseline hits:   {broke}")

    # Token cost per call
    p = [r[4] for r in ok]
    c = [r[5] for r in ok]
    print(f"\ntokens per call: prompt avg {sum(p) / len(p):.0f}, "
          f"completion avg {sum(c) / len(c):.0f}, max total {max(a + b for a, b in zip(p, c))}")


if __name__ == "__main__":
    main()