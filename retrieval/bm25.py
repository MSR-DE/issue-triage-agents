"""BM25 keyword search over past issues.

How BM25 scores one issue against a query, in plain words. For every query word
that appears in the issue, add:
  - how RARE the word is across all issues (IDF): "musllinux" counts far more than "install"
  - times how often it appears in this issue, with diminishing returns
    (the 10th "error" adds much less than the 1st)
  - adjusted for length, so a 500-line log doesn't match everything just by being long.
rank_bm25 does the math; we give it the tokens.
"""
import numpy as np
from rank_bm25 import BM25Okapi

from retrieval.text import issue_text, tokenize

LOAD_ISSUES = """
SELECT issue_number, title, body, created_at
FROM issues
WHERE repo = %s AND user_type <> 'Bot'
ORDER BY issue_number
"""


class BM25Index:
    def __init__(self, conn, repo, stem=False):
        self.stem = stem
        rows = conn.execute(LOAD_ISSUES, (repo,)).fetchall()
        # numpy arrays, so we can filter all 9k issues in one step instead of a Python loop
        self.numbers = np.array([r[0] for r in rows])
        self.created = np.array([r[3].timestamp() for r in rows])   # datetime -> seconds since 1970
        corpus = [tokenize(issue_text(title, body), self.stem) for _, title, body, _ in rows]
        # Builds word counts and IDF over ALL issues. Note: IDF also "sees" issues created
        # later than a given query. That leaks only word-rarity statistics, never answers,
        # because search() below never returns a later issue.
        self.bm25 = BM25Okapi(corpus)

    def search(self, query, before, k=10):
        """Top-k issue numbers for `query`, only among issues created before `before`."""
        scores = self.bm25.get_scores(tokenize(query, self.stem)) # one score per issue, same order as self.numbers
        allowed = self.created < before.timestamp()               # True/False per issue: no peeking at the future
        scores = np.where(allowed, scores, -np.inf)               # banned issues can never rank
        top = np.argsort(-scores)[:k]                             # positions of the k highest scores
        return [int(self.numbers[i]) for i in top if scores[i] > -np.inf]