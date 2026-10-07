"""Load pull requests from the /issues pages already stored in Phase A (no new fetch).

GitHub's /issues endpoint returns PRs too (that's why Phase A saw 12,637 of them);
their "pull_request" field carries merged_at. We skipped them for the issues table;
the Investigator needs them: which PRs were merged before a given issue was opened.

Also records "Fixes #N" / "Closes #N" / "Resolves #N" in PR bodies (pr_closes): the
PR's own claim to fix an issue. Used to pick Investigator controls (bugs fixed only
AFTER they were reported). Snapshot as of the Phase A fetch (2 Oct 2026).

    python -m ingestion.prs
"""
import re

from ingestion.db import get_db_connection

REPO = "astral-sh/uv"

UPSERT_PR = """
INSERT INTO pull_requests (repo, pr_number, title, body, user_login, user_type, state,
                           created_at, closed_at, merged_at)
VALUES (%(repo)s, %(pr_number)s, %(title)s, %(body)s, %(user_login)s, %(user_type)s,
        %(state)s, %(created_at)s, %(closed_at)s, %(merged_at)s)
ON CONFLICT (repo, pr_number) DO UPDATE SET
    title = EXCLUDED.title, body = EXCLUDED.body, user_login = EXCLUDED.user_login,
    user_type = EXCLUDED.user_type, state = EXCLUDED.state, created_at = EXCLUDED.created_at,
    closed_at = EXCLUDED.closed_at, merged_at = EXCLUDED.merged_at
"""
DELETE_CLOSES = "DELETE FROM pr_closes WHERE repo = %s AND pr_number = %s"
INSERT_CLOSE = """
INSERT INTO pr_closes (repo, pr_number, issue_number)
SELECT %s, %s, %s
WHERE EXISTS (SELECT 1 FROM issues WHERE repo = %s AND issue_number = %s)
ON CONFLICT DO NOTHING
"""

# GitHub's closing keywords followed by an issue reference.
CLOSES = re.compile(
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s+"
    r"(?:https://github\.com/astral-sh/uv/issues/|astral-sh/uv#|#)(\d+)",
    re.I,
)


def extract_pr(item):
    """A PR row from an /issues item, or None if the item is an issue."""
    if "pull_request" not in item:
        return None
    user = item.get("user") or {}          # deleted accounts come back as null
    return {
        "pr_number": item["number"],
        "title": item["title"],
        "body": item["body"],
        "user_login": user.get("login"),
        "user_type": user.get("type"),
        "state": item["state"],
        "created_at": item["created_at"],
        "closed_at": item["closed_at"],
        "merged_at": item["pull_request"].get("merged_at"),
    }


def closed_issues(body):
    return sorted({int(n) for n in CLOSES.findall(body or "")})


def load_prs():
    with get_db_connection() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM raw_responses WHERE repo = %s AND kind = 'issues_page' "
            "AND http_status = 200 ORDER BY id", (REPO,)).fetchall()]
        n_prs = n_merged = 0
        for raw_id in ids:
            items = conn.execute("SELECT body FROM raw_responses WHERE id = %s", (raw_id,)).fetchone()[0]
            for item in items:
                pr = extract_pr(item)
                if pr is None:
                    continue
                conn.execute(UPSERT_PR, {**pr, "repo": REPO})
                conn.execute(DELETE_CLOSES, (REPO, pr["pr_number"]))
                for n in closed_issues(pr["body"]):
                    if n != pr["pr_number"]:
                        conn.execute(INSERT_CLOSE, (REPO, pr["pr_number"], n, REPO, n))
                n_prs += 1
                n_merged += pr["merged_at"] is not None
            conn.commit()
        n_closes = conn.execute("SELECT count(*) FROM pr_closes WHERE repo = %s", (REPO,)).fetchone()[0]
        print(f"{n_prs} PRs ({n_merged} merged), {n_closes} 'fixes #N' links to issues")


if __name__ == "__main__":
    load_prs()
