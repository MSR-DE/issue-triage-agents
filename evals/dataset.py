import csv
from pathlib import Path

REPO = "astral-sh/uv"
TYPE_LABELS = ("bug", "enhancement", "question", "documentation")

# Test set: the 300 most recent non-bot issues, keeping those with exactly one type label.
TEST_SQL = """
WITH recent AS (
    SELECT issue_number, title, body, created_at
    FROM issues
    WHERE repo = %(repo)s AND user_type <> 'Bot'
    ORDER BY created_at DESC
    LIMIT 300
)
SELECT r.issue_number, r.title, r.body, MIN(l.label_name) AS label
FROM recent r
JOIN issue_labels l
  ON l.repo = %(repo)s
 AND l.issue_number = r.issue_number
 AND l.label_name = ANY(%(labels)s)
GROUP BY r.issue_number, r.title, r.body, r.created_at
HAVING COUNT(*) = 1
ORDER BY r.created_at
"""

# Dev set: the 50 most recent single-label non-bot issues created before the test window.
DEV_SQL = """
WITH cutoff AS (
    SELECT MIN(created_at) AS t
    FROM (
        SELECT created_at
        FROM issues
        WHERE repo = %(repo)s AND user_type <> 'Bot'
        ORDER BY created_at DESC
        LIMIT 300
    ) recent
)
SELECT i.issue_number, i.title, i.body, MIN(l.label_name) AS label
FROM issues i
JOIN issue_labels l
  ON l.repo = i.repo
 AND l.issue_number = i.issue_number
 AND l.label_name = ANY(%(labels)s)
WHERE i.repo = %(repo)s
  AND i.user_type <> 'Bot'
  AND i.created_at < (SELECT t FROM cutoff)
GROUP BY i.issue_number, i.title, i.body, i.created_at
HAVING COUNT(*) = 1
ORDER BY i.created_at DESC
LIMIT 50
"""


def _load(conn, sql):
    params = {"repo": REPO, "labels": list(TYPE_LABELS)}
    return conn.execute(sql, params).fetchall()


HERE = Path(__file__).parent

PINNED_SQL = """
SELECT issue_number, title, body
FROM issues
WHERE repo = %(repo)s AND issue_number = ANY(%(numbers)s)
ORDER BY created_at
"""


def _load_pinned(conn, filename):
    with (HERE / filename).open(newline="") as f:
        gold = {int(r["issue_number"]): r["label"] for r in csv.DictReader(f)}
    rows = conn.execute(PINNED_SQL, {"repo": REPO, "numbers": list(gold)}).fetchall()
    if len(rows) != len(gold):
        raise RuntimeError(f"{filename}: {len(gold)} pinned, {len(rows)} found in issues")
    return [(n, title, body, gold[n]) for n, title, body in rows]


def load_test_set(conn):
    return _load_pinned(conn, "test_issues.csv")


def load_dev_set(conn):
    return _load_pinned(conn, "dev_issues.csv")