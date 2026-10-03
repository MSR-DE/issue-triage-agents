"""Phase B: fetch issue timelines for the issues the duplicate and missing-info evals need.

Targets: issues labelled duplicate or needs-mre, or closed with state_reason 'duplicate'.
Each timeline page is stored in raw_responses (kind 'timeline'). All pages of one issue
are committed together, so "first page stored with 200" means the issue is done and a
re-run skips it. Run from the project root:  python -m ingestion.timelines
"""
import time

from psycopg.types.json import Jsonb

from ingestion.db import get_db_connection
from ingestion.github_client import get

REPO = "astral-sh/uv"

TARGETS = """
SELECT DISTINCT i.issue_number
FROM issues i
LEFT JOIN issue_labels l
  ON l.repo = i.repo AND l.issue_number = i.issue_number
WHERE i.repo = %s
  AND (l.label_name IN ('duplicate', 'needs-mre') OR i.state_reason = 'duplicate')
ORDER BY i.issue_number
"""

ALREADY_DONE = """
SELECT url FROM raw_responses
WHERE repo = %s AND kind = 'timeline' AND http_status = 200
"""

SAVE_RAW = """
INSERT INTO raw_responses (repo, url, kind, http_status, body)
VALUES (%s, %s, 'timeline', %s, %s)
ON CONFLICT (url) DO UPDATE SET
    http_status = EXCLUDED.http_status,
    body        = EXCLUDED.body,
    fetched_at  = now()
"""


def first_page_url(issue_number):
    return f"https://api.github.com/repos/{REPO}/issues/{issue_number}/timeline?per_page=100"


def fetch_timelines():
    with get_db_connection() as conn:
        targets = [r[0] for r in conn.execute(TARGETS, (REPO,)).fetchall()]
        done = {r[0] for r in conn.execute(ALREADY_DONE, (REPO,)).fetchall()}
        todo = [n for n in targets if first_page_url(n) not in done]
        print(f"{len(targets)} target issues, {len(targets) - len(todo)} already fetched, {len(todo)} to go")

        for i, number in enumerate(todo, 1):
            url = first_page_url(number)
            pages = 0
            while url:
                status, body, next_url = get(url)
                if status != 200:
                    conn.rollback()      # drop this issue's earlier pages so it counts as not done
                    conn.execute(SAVE_RAW, (REPO, url, status, Jsonb(body)))
                    conn.commit()        # keep only the error row; a re-run retries the issue
                    print(f"Stopped: HTTP {status} on {url}")
                    return False
                conn.execute(SAVE_RAW, (REPO, url, status, Jsonb(body)))
                pages += 1
                url = next_url
                time.sleep(0.3)
            conn.commit()                # all pages of this issue land together
            print(f"[{i}/{len(todo)}] #{number}: {pages} page(s)")

        return True


if __name__ == "__main__":
    fetch_timelines()
