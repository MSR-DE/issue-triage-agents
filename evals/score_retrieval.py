"""Retrieval eval: for each known duplicate, does search find the original?

Answer key: duplicate_links (built by `python -m ingestion.duplicates`).
For each duplicate we search with ITS title + the start of its body, exactly what the
bot would see when the issue is first opened, among issues created before it.
hit@k = at least one maintainer-linked original is in the top k results.

Run: python -m evals.score_retrieval
"""
from collections import defaultdict

from evals.dataset import REPO
from ingestion.db import get_db_connection
from retrieval.bm25 import BM25Index
from retrieval.text import issue_text
from retrieval.hybrid import HybridIndex
from retrieval.vector import VectorIndex

KS = (1, 5, 10)
QUERY_BODY_CHARS = 1500    # same cap as the labeler and the planned embeddings

# One row per duplicate: its tier, the list of originals (array_agg), and its own text.
CASES = """
SELECT d.issue_number, d.tier, array_agg(d.original_number), i.title, i.body, i.created_at
FROM duplicate_links d
JOIN issues i ON i.repo = d.repo AND i.issue_number = d.issue_number
WHERE d.repo = %s
GROUP BY d.issue_number, d.tier, i.title, i.body, i.created_at
ORDER BY d.issue_number
"""


def evaluate(index, cases):
    """Run every duplicate through one index. Returns (totals, hits, misses)."""
    totals = defaultdict(int)                       # totals[group] = number of duplicates
    hits = defaultdict(lambda: defaultdict(int))    # hits[group][k] = how many had an original in the top k
    misses = []                                     # not found even in the top 10

    for number, tier, originals, title, body, created_at in cases:
        query = issue_text(title, body, QUERY_BODY_CHARS)
        results = index.search(query, before=created_at, k=max(KS))

        # Each duplicate counts in its own tier; explicit + single_link also count in "headline".
        groups = [tier] + (["headline"] if tier in ("explicit", "single_link") else [])
        for g in groups:
            totals[g] += 1
            for k in KS:
                if set(results[:k]) & set(originals):    # & = overlap between the two sets
                    hits[g][k] += 1
        if not set(results) & set(originals):
            misses.append(number)
    return totals, hits, misses


def print_report(name, totals, hits, misses):
    print(f"\n== {name} ==")
    print(f"{'group':12} {'n':>4}" + "".join(f"  recall@{k:<3}" for k in KS))
    for g in ("headline", "explicit", "single_link", "multi_link"):
        n = totals[g]
        print(f"{g:12} {n:4d}" + "".join(f"  {hits[g][k] / n:9.1%}" for k in KS))
    print(f"missed even at top {max(KS)}: {len(misses)}, e.g. {misses[:10]}")


def main():
    with get_db_connection() as conn:
        cases = conn.execute(CASES, (REPO,)).fetchall()
        bm25 = BM25Index(conn, REPO)          # plain BM25: stemming didn't help (23.0% vs 24.7% recall@5)
        vector = VectorIndex(conn, REPO)

        # All three have the same .search(query, before, k), so evaluate() scores them identically.
        indexes = {
            "BM25": bm25,
            "Vector (bge-small)": vector,
            "Hybrid (BM25 + vector, RRF)": HybridIndex(bm25, vector),
        }

        # Stay INSIDE the `with` block: VectorIndex runs one SQL query per search,
        # so the database connection must still be open while we evaluate.
        for name, index in indexes.items():
            print_report(name, *evaluate(index, cases))   # same 344 duplicates for every method

if __name__ == "__main__":
    main()