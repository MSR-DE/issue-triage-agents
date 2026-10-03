"""Hybrid search: combine BM25 (exact words) and vector search (meaning) with
Reciprocal Rank Fusion.

RRF ignores the raw scores (BM25 scores and cosine distances are on different scales)
and uses only each result's RANK in each list: score = sum of 1 / (60 + rank).
An issue ranked 3rd by BM25 and 5th by vectors beats one ranked 1st by only one of them.
60 is the standard constant from the original RRF paper; it stops the very top ranks
from completely dominating.
"""
from collections import defaultdict

RRF_K = 60
DEPTH = 50   # take the top 50 from each list before fusing, so an issue ranked 20th
             # in one list and 3rd in the other still gets a chance to rise


def rrf(rankings, k=RRF_K):
    scores = defaultdict(float)
    for ranking in rankings:                         # one ranked list per search method
        for rank, number in enumerate(ranking, 1):   # rank starts at 1
            scores[number] += 1 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)   # best fused score first


class HybridIndex:
    def __init__(self, *indexes):
        self.indexes = indexes                       # e.g. (bm25_index, vector_index)

    def search(self, query, before, k=10):
        rankings = [ix.search(query, before, DEPTH) for ix in self.indexes]
        return rrf(rankings)[:k]