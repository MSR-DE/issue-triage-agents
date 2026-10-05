"""Retrieval eval: for each known duplicate, does search find the original?

Answer key: duplicate_links (built by `python -m ingestion.duplicates`).
For each duplicate we search with ITS title + the start of its body, exactly what the
bot would see when the issue is first opened, among issues created before it.
hit@k = at least one maintainer-linked original is in the top k results (strict).
Cluster-aware hit@k also counts any other issue maintainers linked to the same
original (evals/dup_dataset.py); both are reported side by side.

Run: python -m evals.score_retrieval
"""
from collections import defaultdict
from math import comb   # comb(n, i) = "n choose i", used for the exact McNemar test

from evals.dataset import REPO
from evals.dup_dataset import accepted, load_clusters
from ingestion.db import get_db_connection
from retrieval.bm25 import BM25Index
from retrieval.hybrid import HybridIndex
from retrieval.text import issue_text
from retrieval.vector import VectorIndex

KS = (1, 5, 10)
QUERY_BODY_CHARS = 1500    # same cap as the labeler and the embeddings

# One row per duplicate: its tier, the list of originals (array_agg), and its own text.
CASES = """
SELECT d.issue_number, d.tier, array_agg(d.original_number), i.title, i.body, i.created_at
FROM duplicate_links d
JOIN issues i ON i.repo = d.repo AND i.issue_number = d.issue_number
WHERE d.repo = %s
GROUP BY d.issue_number, d.tier, i.title, i.body, i.created_at
ORDER BY d.issue_number
"""


def evaluate(index, cases, clusters):
    """Run every duplicate through one index.
    Returns (totals, hits, hits_c, misses, found5, found5_c); *_c = cluster-aware."""
    totals = defaultdict(int)                       # totals[group] = number of duplicates
    hits = defaultdict(lambda: defaultdict(int))    # hits[group][k] = how many had an original in the top k
    hits_c = defaultdict(lambda: defaultdict(int))  # same, counting cluster siblings too
    misses = []                                     # not found even in the top 10
    found5 = set()                                  # headline duplicates found in the top 5 (for fixed vs broke)
    found5_c = set()

    for number, tier, originals, title, body, created_at in cases:
        query = issue_text(title, body, QUERY_BODY_CHARS)
        results = index.search(query, before=created_at, k=max(KS))

        ok = accepted(number, set(originals), clusters)
        if tier in ("explicit", "single_link") and set(results[:5]) & set(originals):
            found5.add(number)
        if tier in ("explicit", "single_link") and set(results[:5]) & ok:
            found5_c.add(number)

        # Each duplicate counts in its own tier; explicit + single_link also count in "headline".
        groups = [tier] + (["headline"] if tier in ("explicit", "single_link") else [])
        for g in groups:
            totals[g] += 1
            for k in KS:
                if set(results[:k]) & set(originals):    # & = overlap between the two sets
                    hits[g][k] += 1
                if set(results[:k]) & ok:
                    hits_c[g][k] += 1
        if not set(results) & set(originals):
            misses.append(number)
    return totals, hits, hits_c, misses, found5, found5_c


def print_report(name, totals, hits, misses, hits_c):
    print(f"\n== {name} ==   (strict | cluster)")
    print(f"{'group':12} {'n':>4}" + "".join(f"  recall@{k:<3}" for k in KS)
          + "  |" + "".join(f"  recall@{k:<3}" for k in KS))
    for g in ("headline", "explicit", "single_link", "multi_link"):
        n = totals[g]
        print(f"{g:12} {n:4d}" + "".join(f"  {hits[g][k] / n:9.1%}" for k in KS)
              + "  |" + "".join(f"  {hits_c[g][k] / n:9.1%}" for k in KS))
    print(f"missed even at top {max(KS)}: {len(misses)}, e.g. {misses[:10]}")


def compare(name_a, found_a, name_b, found_b):
    """Paired comparison on the same duplicates, like the labeler's fixed vs broke.
    Only duplicates where the two methods disagree tell us which is better."""
    fixed = found_b - found_a          # B finds it in the top 5, A doesn't
    broke = found_a - found_b          # A finds it, B doesn't
    n = len(fixed) + len(broke)
    # Exact McNemar: if B were no better than A, each disagreement would be a coin flip.
    # p = chance of a split at least this lopsided from fair coins (two-sided).
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(len(fixed), len(broke)) + 1)) / 2 ** n) if n else 1.0
    print(f"{name_b} vs {name_a}: fixed {len(fixed)}, broke {len(broke)}, p = {p:.4f}")


def main():
    with get_db_connection() as conn:
        cases = conn.execute(CASES, (REPO,)).fetchall()
        clusters = load_clusters(conn)
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
        found, found_c = {}, {}
        for name, index in indexes.items():
            totals, hits, hits_c, misses, found[name], found_c[name] = evaluate(index, cases, clusters)
            print_report(name, totals, hits, misses, hits_c)

        for label, f in (("strict", found), ("cluster", found_c)):
            print(f"\n== paired, headline duplicates, top 5, {label} ==")
            compare("BM25", f["BM25"], "Vector (bge-small)", f["Vector (bge-small)"])
            compare("Vector (bge-small)", f["Vector (bge-small)"], "Hybrid (BM25 + vector, RRF)", f["Hybrid (BM25 + vector, RRF)"])


if __name__ == "__main__":
    main()