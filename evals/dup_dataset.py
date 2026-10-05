"""Load the pinned Duplicate Finder samples (evals/dup_dev.csv, dup_test.csv)."""
import csv
import sys
from pathlib import Path

from evals.dataset import REPO
from ingestion.duplicates import META_ISSUES

HERE = Path(__file__).parent

ISSUES = """
SELECT issue_number, title, body, created_at
FROM issues
WHERE repo = %s AND issue_number = ANY(%s)
"""


def load_dup_set(conn, split):
    """[(number, title, body, created_at, kind, originals)], kind = duplicate | control.
    originals is a set of issue numbers (empty for controls). Meta issues are removed
    from originals (the CSVs were pinned before that rule); a duplicate left with no
    original can't be scored and is skipped, with a note."""
    with (HERE / f"dup_{split}.csv").open(newline="") as f:
        pinned = {int(r["issue_number"]): (r["kind"], {int(x) for x in r["originals"].split(";") if x})
                  for r in csv.DictReader(f)}
    for n, (kind, originals) in list(pinned.items()):
        originals -= META_ISSUES
        if kind == "duplicate" and not originals:
            print(f"note: dup_{split}.csv #{n} skipped (its only original was a meta issue)",
                  file=sys.stderr)
            del pinned[n]
    rows = conn.execute(ISSUES, (REPO, list(pinned))).fetchall()
    if len(rows) != len(pinned):
        raise RuntimeError(f"dup_{split}.csv: {len(pinned)} pinned, {len(rows)} found in issues")
    return [(n, title, body, created_at, *pinned[n])
            for n, title, body, created_at in sorted(rows)]


# Cluster-aware scoring. Maintainers often point several duplicates at one tracking
# issue, so a correct sibling (e.g. #1374 for #1761, both tied to #1526 via #1705)
# would count as a miss under the strict key. Clusters = connected components of
# maintainer duplicate links, explicit + single_link only (multi_link includes vaguer
# "related" mentions that could chain unrelated issues together).
EDGES = """
SELECT issue_number, original_number
FROM duplicate_links
WHERE repo = %s AND tier IN ('explicit', 'single_link')
  AND original_number <> ALL(%s)      -- meta issues never join clusters
"""


def load_clusters(conn):
    """{issue_number: frozenset of every issue in its cluster} (union-find)."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in conn.execute(EDGES, (REPO, list(META_ISSUES))).fetchall():
        parent[find(a)] = find(b)

    members = {}
    for x in list(parent):
        members.setdefault(find(x), set()).add(x)
    return {x: frozenset(members[find(x)]) for x in parent}


def accepted(number, originals, clusters):
    """Strict originals plus every other member of their clusters (not the issue itself)."""
    ok = set(originals)
    for o in originals:
        ok |= clusters.get(o, frozenset())
    ok.discard(number)
    return ok
