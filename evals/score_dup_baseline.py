"""Plain vector search on the pinned Duplicate Finder samples: the score the
agent has to beat. No LLM tokens (local embeddings + Postgres).

    python -m evals.score_dup_baseline --split dev

Duplicates: recall@5 / @10 (any maintainer-linked original in the top k), same
query as score_retrieval (title + first 1,500 chars of cleaned body).
Cluster recall also counts any other issue maintainers linked to the same original
(see evals/dup_dataset.py). Controls: plain search always returns k results, so it can never say "no
duplicate"; the agent's controls score is measured against that.
Also prints the size of a 10-candidate search result, for the token budget.
"""
import argparse

from ingestion.db import get_db_connection
from evals.dataset import REPO
from evals.dup_dataset import accepted, load_clusters, load_dup_set
from retrieval.lookup import search_candidates
from retrieval.text import issue_text

QUERY_BODY_CHARS = 1500


def main(index_factory=None):
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["dev", "test", "test_small"], required=True)
    args = p.parse_args()

    with get_db_connection() as conn:
        if index_factory is None:
            from retrieval.vector import VectorIndex      # loads the embedding model
            index = VectorIndex(conn, REPO)
        else:
            index = index_factory(conn)
        cases = load_dup_set(conn, args.split)
        clusters = load_clusters(conn)

        hits5, hits10, n_dup, n_ctl, sizes = [], 0, 0, 0, []
        c_hits5, c_hits10 = [], 0
        for number, title, body, created_at, kind, originals in cases:
            cands = search_candidates(conn, index, REPO, issue_text(title, body, QUERY_BODY_CHARS),
                                      before=created_at, k=10)
            sizes.append(sum(len(f"#{c['number']} {c['title']}: {c['snippet']}") for c in cands))
            if kind == "control":
                n_ctl += 1
                continue
            n_dup += 1
            top = [c["number"] for c in cands]
            if set(top[:5]) & originals:
                hits5.append(number)
            if set(top) & originals:
                hits10 += 1
            ok = accepted(number, originals, clusters)
            if set(top[:5]) & ok:
                c_hits5.append(number)
            if set(top) & ok:
                c_hits10 += 1

    print(f"split {args.split}: {n_dup} duplicates, {n_ctl} controls")
    print(f"vector recall@5:  {len(hits5)}/{n_dup} = {len(hits5) / n_dup:.1%}")
    print(f"vector recall@10: {hits10}/{n_dup} = {hits10 / n_dup:.1%}")
    print(f"found in top 5: {sorted(hits5)}")
    print(f"cluster recall@5:  {len(c_hits5)}/{n_dup} = {len(c_hits5) / n_dup:.1%}"
          f"  (also counts other issues maintainers linked to the same original)")
    print(f"cluster recall@10: {c_hits10}/{n_dup} = {c_hits10 / n_dup:.1%}")
    print(f"found only via a cluster sibling: {sorted(set(c_hits5) - set(hits5))}")
    print(f"controls: plain search returns 10 candidates for all {n_ctl} (can't say 'none')")
    avg = sum(sizes) / len(sizes)
    print(f"10-candidate result: ~{avg:.0f} chars (~{avg / 4:.0f} tokens) on average")


if __name__ == "__main__":
    main()
