"""Score an Investigator run against plain vector search over merged PRs (same index and
same search window as the agent), on the same issues.

    python -m evals.score_inv_run inveval-20b-v1

Already-fixed issues: hit = a maintainer-linked fixing PR is in the answer. Agent vs
vector top 5, and same-length (vector's top n, n = the agent's list length), with
fixed/broke + exact McNemar; precision of suggestions. Controls (not fixed when
reported): how often the agent correctly says "not fixed" (plain search never can).
Decision accuracy: fixed issues answered with a right PR + controls answered "not fixed".
Plus invented numbers, rounds, reads, tokens. The eval set is small (~30 + 30), so
rates come with Wilson 95% intervals: report the range, not just the point.
"""
import sys
from collections import Counter
from math import sqrt

from ingestion.db import get_db_connection
from evals.dataset import REPO
from evals.fix_dataset import load_fix_set
from evals.score_dup_run import mcnemar
from retrieval.fix_lookup import reported_version, search_fix_candidates
from retrieval.text import issue_text
from triage.investigator import QUERY_BODY_CHARS, make_index, search_since

def ci(k, n):
    """'k/n = p% (95% CI lo-hi%)', Wilson score interval (sane at small n and at 0 or n)."""
    if not n:
        return "0/0"
    z, p = 1.96, k / n
    mid = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{k}/{n} = {p:.0%} (95% CI {max(0, mid - half):.0%}-{min(1, mid + half):.0%})"


ROWS = """
SELECT issue_number, split, returned, invented, rounds, queries, reads,
       prompt_tokens, completion_tokens, error
FROM inv_eval_results WHERE run_id = %s
"""


def main(run_id, index_factory=None):
    with get_db_connection() as conn:
        rows = {r[0]: r for r in conn.execute(ROWS, (run_id,)).fetchall()}
        if not rows:
            raise SystemExit(f"no rows for run {run_id}")
        split = next(iter(rows.values()))[1]
        cases = [c for c in load_fix_set(conn, split) if c[0] in rows]
        index = (index_factory or make_index)(conn)
        vector_top5 = {
            num: [c["number"] for c in search_fix_candidates(
                conn, index, REPO, issue_text(title, body, QUERY_BODY_CHARS), before=created_at,
                reported=reported_version(body), k=5,
                since=search_since(conn, created_at, reported_version(body)))]
            for num, title, body, created_at, _, _ in cases
        }

    failed = [n for n, r in rows.items() if r[2] is None]
    ok = [c for c in cases if rows[c[0]][2] is not None]
    fixed = [c for c in ok if c[4] == "fixed"]
    ctls = [c for c in ok if c[4] == "control"]
    print(f"run {run_id} ({split}): {len(fixed)} already-fixed, {len(ctls)} controls, "
          f"{len(failed)} failed {failed or ''}")

    if fixed:
        agent_hit = {n for n, *_, prs in fixed if set(rows[n][2]) & prs}
        vec_hit = {n for n, *_, prs in fixed if set(vector_top5[n]) & prs}
        f_, b_ = agent_hit - vec_hit, vec_hit - agent_hit
        print(f"\nfixing PR found:  agent {ci(len(agent_hit), len(fixed))}"
              f"   vector top 5 {ci(len(vec_hit), len(fixed))}")
        print(f"  agent fixed {len(f_)} {sorted(f_)}, broke {len(b_)} {sorted(b_)}, p = {mcnemar(f_, b_):.4f}")
        vec_n = {n for n, *_, prs in fixed if set(vector_top5[n][:max(1, len(rows[n][2]))]) & prs}
        f_, b_ = agent_hit - vec_n, vec_n - agent_hit
        print(f"  same-length: agent {len(agent_hit)}/{len(fixed)}, vector top-n {len(vec_n)}/{len(fixed)}; "
              f"fixed {len(f_)} {sorted(f_)}, broke {len(b_)} {sorted(b_)}, p = {mcnemar(f_, b_):.4f}")
        shown = sum(len(rows[n][2]) for n, *_ in fixed)
        right = sum(len(set(rows[n][2]) & prs) for n, *_, prs in fixed)
        right_v = sum(len(set(vector_top5[n]) & prs) for n, *_, prs in fixed)
        shown_v = sum(len(vector_top5[n]) for n, *_ in fixed)
        print(f"  precision of suggestions: agent {right}/{shown} = {right / max(1, shown):.0%}, "
              f"vector top 5 {right_v}/{shown_v} = {right_v / max(1, shown_v):.0%}")
        missed = [n for n, *_ in fixed if not rows[n][2]]
        print(f"  said 'not fixed' on {len(missed)} already-fixed issues {missed}")

    if ctls:
        empty = [n for n, *_ in ctls if not rows[n][2]]
        flagged = {n: rows[n][2] for n, *_ in ctls if rows[n][2]}
        print(f"\ncontrols: said 'not fixed' on {ci(len(empty), len(ctls))} "
              f"(vector: 0%); false alarms: {flagged}")

    right = (sum(1 for n, *_, prs in fixed if set(rows[n][2]) & prs)
             + sum(1 for n, *_ in ctls if not rows[n][2]))
    print(f"\ndecision accuracy (right PR on fixed, 'not fixed' on controls): "
          f"{ci(right, len(ok))}")

    done = [rows[n] for n, *_ in ok]
    print(f"invented numbers (dropped): {sum(len(r[3]) for r in done)}")
    print(f"rounds: {dict(sorted(Counter(r[4] for r in done).items()))}; "
          f"searched again on {sum(1 for r in done if r[5])}/{len(done)}; "
          f"read a PR on {sum(1 for r in done if r[6])}/{len(done)}")
    totals = [r[7] + r[8] for r in done]
    print(f"tokens per issue: avg {sum(totals) / len(totals):.0f}, max {max(totals)}")


if __name__ == "__main__":
    main(sys.argv[1])
