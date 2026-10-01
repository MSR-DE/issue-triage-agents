import time
from dotenv import load_dotenv
from psycopg.types.json import Jsonb
from ingestion.db import get_db_connection
from ingestion.github_client import get

REPO = "astral-sh/uv"

START_URL = f"https://api.github.com/repos/{REPO}/issues?state=all&per_page=100"  ## 100 is the max per page limit ig



# Save one fetched page into raw_responses.
# url is UNIQUE, so ON CONFLICT turns a duplicate insert into an update.
# That is what makes re-running the fetch idempotent.
# EXCLUDED = the row we just tried to insert.
SAVE_RAW = """
INSERT INTO raw_responses (repo, url, kind, http_status, body)
VALUES (%s, %s, 'issues_page', %s, %s)
ON CONFLICT (url) DO UPDATE SET
    http_status = EXCLUDED.http_status,
    body        = EXCLUDED.body,
    fetched_at  = now()
"""

# Remember where we stopped so the loop can resume.
# One row per repo (repo is the primary key); each page updates it.
# next_url is None once the last page is fetched.
SAVE_STATE = """
INSERT INTO sync_state (repo, phase, next_url, status, updated_at)
VALUES (%s, 'A', %s, %s, now())
ON CONFLICT (repo) DO UPDATE SET
    next_url   = EXCLUDED.next_url,
    status     = EXCLUDED.status,
    updated_at = now()
"""



def fetch_all():
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT next_url, status FROM sync_state WHERE repo = %s", (REPO,)
        ).fetchone()

        url = START_URL
        if row:
            saved_next, status = row
            if status == "done":
                print("Already finished.")
                return
            url = saved_next

        pages = 0
        while url:
            status, body, next_url = get(url)
            conn.execute(SAVE_RAW, (REPO, url, status, Jsonb(body)))

            if status != 200:
                conn.commit()
                print(f"Stopped: HTTP {status} on {url}")
                break

            conn.execute(
                SAVE_STATE,
                (REPO, next_url, "done" if next_url is None else "running"),
            )
            conn.commit()

            pages += 1
            print(f"page {pages}: {len(body)} items")
            url = next_url
            time.sleep(0.3)


if __name__ == "__main__":
    fetch_all()