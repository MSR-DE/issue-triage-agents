"""Fetch every issue/PR comment in the repo (needed for the "already fixed?" answer key).

One repo-wide endpoint (/issues/comments) instead of one request per issue: ~100
comments per request, so ~700-900 requests for uv (a few minutes with the token).

Cursor: GitHub refuses page numbers past 300 on this endpoint, so we page by time:
sort by updated_at ascending and ask for comments updated since the newest one we
have. Each page is stored raw and its comments upserted in the same transaction, so
the cursor is simply max(updated_at) in issue_comments: Ctrl+C and re-run resumes.
`since` is inclusive, so the boundary comment repeats once per page (upsert, harmless).
If 100+ comments share one timestamp, we step through them with page=2, 3, ...

    python -m ingestion.comments
"""
import time
from datetime import timezone
from urllib.parse import quote

from psycopg.types.json import Jsonb

from ingestion.db import get_db_connection
from ingestion.github_client import get

REPO = "astral-sh/uv"
BASE = f"https://api.github.com/repos/{REPO}/issues/comments?sort=updated&direction=asc&per_page=100"
EPOCH = "2023-01-01T00:00:00Z"        # before uv's first issue
PER_PAGE = 100

SAVE_RAW = """
INSERT INTO raw_responses (repo, url, kind, http_status, body)
VALUES (%s, %s, 'comments_page', %s, %s)
ON CONFLICT (url) DO UPDATE SET
    http_status = EXCLUDED.http_status,
    body        = EXCLUDED.body,
    fetched_at  = now()
"""
UPSERT_COMMENT = """
INSERT INTO issue_comments (repo, comment_id, issue_number, user_login, user_type,
                            author_association, created_at, updated_at, body, html_url)
VALUES (%(repo)s, %(comment_id)s, %(issue_number)s, %(user_login)s, %(user_type)s,
        %(author_association)s, %(created_at)s, %(updated_at)s, %(body)s, %(html_url)s)
ON CONFLICT (repo, comment_id) DO UPDATE SET
    author_association = EXCLUDED.author_association,
    updated_at = EXCLUDED.updated_at, body = EXCLUDED.body
"""
CURSOR = "SELECT max(updated_at) FROM issue_comments WHERE repo = %s"


def extract_comment(c):
    user = c.get("user") or {}
    return {
        "comment_id": c["id"],
        "issue_number": int(c["issue_url"].rsplit("/", 1)[1]),
        "user_login": user.get("login"),
        "user_type": user.get("type"),
        "author_association": c.get("author_association"),
        "created_at": c["created_at"],
        "updated_at": c["updated_at"],
        "body": c.get("body"),
        "html_url": c.get("html_url"),
    }


def iso(ts):
    # psycopg returns timestamptz in the session's time zone; GitHub wants UTC ("Z").
    return ts.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if ts else EPOCH


def fetch_comments():
    with get_db_connection() as conn:
        since = iso(conn.execute(CURSOR, (REPO,)).fetchone()[0])
        page, total = 1, 0
        while True:
            url = f"{BASE}&since={quote(since)}" + (f"&page={page}" if page > 1 else "")
            status, body, _ = get(url)
            conn.execute(SAVE_RAW, (REPO, url, status, Jsonb(body)))
            if status != 200:
                conn.commit()
                print(f"Stopped: HTTP {status} on {url}")
                return False
            for c in body:
                conn.execute(UPSERT_COMMENT, {**extract_comment(c), "repo": REPO})
            conn.commit()
            total += len(body)
            if len(body) < PER_PAGE:
                print(f"done: {total} comments fetched this run")
                return True
            newest = max(c["updated_at"] for c in body)
            if newest == since:
                page += 1                  # a whole page with one timestamp: step through it
            else:
                since, page = newest, 1
            print(f"{total} comments, up to {since}")
            time.sleep(0.3)


if __name__ == "__main__":
    fetch_comments()
