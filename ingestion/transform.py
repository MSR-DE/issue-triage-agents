
def extract_issue(item):
    # PRs share the /issues endpoint, they carry a "pull_request" field
    if "pull_request" in item:
        return None

    return {
        "issue_number": item["number"],
        "github_id": item["id"],
        "title": item["title"],
        "body": item["body"],                    # can be None, keep it
        "state": item["state"],
        "state_reason": item.get("state_reason"),  # may be missing
        "user_login": item["user"]["login"],
        "user_type": item["user"]["type"],       # "Bot" rows are filtered later
        "created_at": item["created_at"],        # ISO strings; Postgres parses them
        "updated_at": item["updated_at"],
        "closed_at": item["closed_at"],          # None while open
        "labels": [label["name"] for label in item["labels"]],
    }


if __name__ == "__main__":
    from ingestion.db import get_db_connection

    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT body FROM raw_responses WHERE http_status = 200 ORDER BY id LIMIT 1"
        ).fetchone()

    items = row[0]  # psycopg already turns jsonb into a Python list
    rows = [r for r in map(extract_issue, items) if r]
    print(len(items), "items ->", len(rows), "issues")
    print(rows[0])