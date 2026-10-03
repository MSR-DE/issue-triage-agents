"""Few-shot examples for the labeler: similar PAST issues and how maintainers labeled them.

An issue may be an example only if all three hold:
  1. it was created before the issue being labeled     (no future knowledge)
  2. it is not inside the test window                  (no test issue sees another test issue's label)
  3. it has exactly one type label                     (one clear answer per example)
Rules 1 and 2 become one cutoff date passed to search(); rule 3 is a dictionary lookup.
"""
import csv

from evals.dataset import HERE, REPO, TYPE_LABELS
from retrieval.text import issue_text

N_EXAMPLES = 5
SEARCH_DEPTH = 30     # ask search for more than 5: some results have no type label (or several)
QUERY_BODY_CHARS = 1500

# Issues with exactly one of our four type labels -> {issue_number: label}
SINGLE_LABEL = """
SELECT issue_number, MIN(label_name)
FROM issue_labels
WHERE repo = %s AND label_name = ANY(%s)
GROUP BY issue_number
HAVING COUNT(*) = 1
"""


class ExampleFinder:
    def __init__(self, conn, index):
        self.index = index                        # BM25Index, VectorIndex or HybridIndex: all have .search()
        self.label = dict(conn.execute(SINGLE_LABEL, (REPO, list(TYPE_LABELS))).fetchall())
        rows = conn.execute(
            "SELECT issue_number, title, created_at FROM issues WHERE repo = %s", (REPO,)
        ).fetchall()
        self.title = {n: t for n, t, _ in rows}
        self.created = {n: c for n, _, c in rows}

        # Start of the test window = creation time of the oldest pinned test issue.
        with (HERE / "test_issues.csv").open(newline="") as f:
            test_numbers = [int(r["issue_number"]) for r in csv.DictReader(f)]
        self.test_start = min(self.created[n] for n in test_numbers)

    def find(self, number, title, body):
        """Up to 5 (issue_number, title, label) examples for the issue being labeled."""
        cutoff = min(self.created[number], self.test_start)     # rules 1 + 2 in one date
        candidates = self.index.search(issue_text(title, body, QUERY_BODY_CHARS), before=cutoff, k=SEARCH_DEPTH)
        picked = [n for n in candidates if n in self.label][:N_EXAMPLES]   # rule 3, keep search order
        return [(n, self.title[n], self.label[n]) for n in picked]