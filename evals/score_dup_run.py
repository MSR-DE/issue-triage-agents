"""Score a Duplicate Finder run against plain vector search on the same issues.

    python -m evals.score_dup_run dupdev-20b-v1

Duplicates: recall@5 strict (maintainer-linked original) and cluster (also other
issues maintainers linked to the same original), agent vs vector, fixed/broke with
exact McNemar; plus a same-length comparison (agent's list vs vector's top n, n =
the agent's list length), since recall@5 rewards listing 5 and the agent lists 1-2. Controls: how often the agent correctly says "no duplicate" (plain
search never can). Plus invented numbers, rounds, tokens.
"""
import sys
from collections import Counter
from math import comb

from ingestion.db import get_db_connection
from evals.dataset import REPO
from evals.dup_dataset import accepted, load_clusters, load_dup_set
from retrieval.lookup import search_candidates
from retrieval.text import issue_text

QUERY_BODY_CHARS = 1500

ROWS = """
SELECT issue_number, split, returned, invented, rounds, queries, reads,
       prompt_tokens, completion_tokens, error
FROM dup_eval_results WHERE run_id = %s
"""


def mcnemar(fixed, broke):
    n = len(fixed) + len(broke)
    if not n:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(len(fixed), len(broke)) + 1)) / 2 ** n)


def main(run_id, index_factory=None):
    with get_db_connection() as conn:
        rows = {r[0]: r for r in conn.execute(ROWS, (run_id,)).fetchall()}
        if not rows:
            raise SystemExit(f"no rows for run {run_id}")
        split = next(iter(rows.values()))[1]
        cases = [c for c in load_dup_set(conn, split) if c[0] in rows]
        clusters = load_clusters(conn)
        if index_factory is None:
            from retrieval.vector import VectorIndex
            index = VectorIndex(conn, REPO)
        else:
            index = index_factory(conn)
        vector_top5 = {
            num: [c["number"] for c in search_candidates(
                conn, index, REPO, issue_text(title, body, QUERY_BODY_CHARS), before=created_at, k=5)]
            for num, title, body, created_at, _, _ in cases
        }

    failed = [n for n, r in rows.items() if r[2] is None]
    ok_cases = [c for c in cases if rows[c[0]][2] is not None]
    dups = [c for c in ok_cases if c[4] == "duplicate"]
    ctls = [c for c in ok_cases if c[4] == "control"]
    print(f"run {run_id} ({split}): {len(dups)} duplicates, {len(ctls)} controls, {len(failed)} failed {failed or ''}")

    for name, key in (("strict", lambda n, o: o), ("cluster", lambda n, o: accepted(n, o, clusters))):
        agent_hit = {n for n, *_, o in dups if set(rows[n][2]) & key(n, o)}
        vec_hit = {n for n, *_, o in dups if set(vector_top5[n]) & key(n, o)}
        fixed, broke = agent_hit - vec_hit, vec_hit - agent_hit
        print(f"\n{name} recall@5:  agent {len(agent_hit)}/{len(dups)} = {len(agent_hit) / len(dups):.1%}"
              f"   vector {len(vec_hit)}/{len(dups)} = {len(vec_hit) / len(dups):.1%}")
        print(f"  agent fixed {len(fixed)} {sorted(fixed)}, broke {len(broke)} {sorted(broke)}, "
              f"p = {mcnemar(fixed, broke):.4f}")

        # Same-length: recall@5 rewards listing 5; the agent usually lists 1-2. Compare
        # its list with vector's top n, n = how many the agent returned (at least 1).
        agent_n = {n for n, *_, o in dups if set(rows[n][2]) & key(n, o)}
        vec_n = {n for n, *_, o in dups
                 if set(vector_top5[n][:max(1, len(rows[n][2]))]) & key(n, o)}
        fixed, broke = agent_n - vec_n, vec_n - agent_n
        shown_agent = sum(len(rows[n][2]) for n, *_ in dups)
        right_agent = sum(len(set(rows[n][2]) & key(n, o)) for n, *_, o in dups)
        print(f"  same-length: agent {len(agent_n)}/{len(dups)}, vector top-n {len(vec_n)}/{len(dups)}; "
              f"fixed {len(fixed)} {sorted(fixed)}, broke {len(broke)} {sorted(broke)}, "
              f"p = {mcnemar(fixed, broke):.4f}")
        print(f"  precision of the agent's suggestions on duplicates: {right_agent}/{shown_agent}"
              f" = {right_agent / max(1, shown_agent):.0%}")

    lens = [len(rows[n][2]) for n, *_ in dups]
    none_on_dup = [n for n, *_ in dups if not rows[n][2]]
    print(f"\nduplicates: avg {sum(lens) / len(lens):.1f} returned; said 'none' on {len(none_on_dup)} {none_on_dup}")
    if ctls:
        empty = [n for n, *_ in ctls if not rows[n][2]]
        flagged = {n: rows[n][2] for n, *_ in ctls if rows[n][2]}
        print(f"controls: said 'none' on {len(empty)}/{len(ctls)} = {len(empty) / len(ctls):.0%} "
              f"(vector: 0%); false alarms: {flagged}")

    ok = [rows[n] for n, *_ in ok_cases]
    invented = sum(len(r[3]) for r in ok)
    print(f"\ninvented numbers (dropped): {invented}")
    print(f"rounds: {dict(sorted(Counter(r[4] for r in ok).items()))}; "
          f"rewrote the query on {sum(1 for r in ok if r[5])}/{len(ok)}; read an issue on {sum(1 for r in ok if r[6])}/{len(ok)}")
    totals = [r[7] + r[8] for r in ok]
    print(f"tokens per issue: avg {sum(totals) / len(totals):.0f}, max {max(totals)}")


if __name__ == "__main__":
    main(sys.argv[1])
