from ingestion.db import get_db_connection
from ingestion.transform import extract_issue

REPO = "astral-sh/uv"

# Insert the issue, or update it if (repo, issue_number) already exists.
# That makes re-running load.py safe (idempotent).
UPSERT_ISSUE = """
INSERT INTO issues (repo, issue_number, github_id, title, body, state, state_reason,
                    user_login, user_type, created_at, updated_at, closed_at,
                    raw_response_id)
VALUES (%(repo)s, %(issue_number)s, %(github_id)s, %(title)s, %(body)s, %(state)s,
        %(state_reason)s, %(user_login)s, %(user_type)s, %(created_at)s,
        %(updated_at)s, %(closed_at)s, %(raw_id)s)
ON CONFLICT (repo, issue_number) DO UPDATE SET
    github_id       = EXCLUDED.github_id,
    title           = EXCLUDED.title,
    body            = EXCLUDED.body,
    state           = EXCLUDED.state,
    state_reason    = EXCLUDED.state_reason,
    user_login      = EXCLUDED.user_login,
    user_type       = EXCLUDED.user_type,
    created_at      = EXCLUDED.created_at,
    updated_at      = EXCLUDED.updated_at,
    closed_at       = EXCLUDED.closed_at,
    raw_response_id = EXCLUDED.raw_response_id
"""

DELETE_LABELS = "DELETE FROM issue_labels WHERE repo = %s AND issue_number = %s"
INSERT_LABEL = "INSERT INTO issue_labels (repo, issue_number, label_name) VALUES (%s, %s, %s)"


def load_all():
    with get_db_connection() as conn:
        # ids only, so we hold one page in memory at a time
        ids = [r[0] for r in conn.execute(
            "SELECT id FROM raw_responses "
            "WHERE kind = 'issues_page' AND http_status = 200 ORDER BY id"
        ).fetchall()]

        total = 0
        for raw_id in ids:
            items = conn.execute(
                "SELECT body FROM raw_responses WHERE id = %s", (raw_id,)
            ).fetchone()[0]

            for item in items:
                row = extract_issue(item)
                if row is None:          # a PR, skip it
                    continue

                conn.execute(UPSERT_ISSUE, {**row, "repo": REPO, "raw_id": raw_id})

                # labels change over time: wipe this issue's labels, re-add current ones
                conn.execute(DELETE_LABELS, (REPO, row["issue_number"]))
                for name in row["labels"]:
                    conn.execute(INSERT_LABEL, (REPO, row["issue_number"], name))
                total += 1

            conn.commit()                # one transaction per page
            print(f"raw page {raw_id}: {total} issues so far")


if __name__ == "__main__":
    load_all()