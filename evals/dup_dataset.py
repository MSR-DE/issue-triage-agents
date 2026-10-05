"""Load the pinned Duplicate Finder samples (evals/dup_dev.csv, dup_test.csv)."""
import csv
from pathlib import Path

from evals.dataset import REPO

HERE = Path(__file__).parent

ISSUES = """
SELECT issue_number, title, body, created_at
FROM issues
WHERE repo = %s AND issue_number = ANY(%s)
"""


def load_dup_set(conn, split):
    """[(number, title, body, created_at, kind, originals)], kind = duplicate | control.
    originals is a set of issue numbers (empty for controls)."""
    with (HERE / f"dup_{split}.csv").open(newline="") as f:
        pinned = {int(r["issue_number"]): (r["kind"], {int(x) for x in r["originals"].split(";") if x})
                  for r in csv.DictReader(f)}
    rows = conn.execute(ISSUES, (REPO, list(pinned))).fetchall()
    if len(rows) != len(pinned):
        raise RuntimeError(f"dup_{split}.csv: {len(pinned)} pinned, {len(rows)} found in issues")
    return [(n, title, body, created_at, *pinned[n])
            for n, title, body, created_at in sorted(rows)]
