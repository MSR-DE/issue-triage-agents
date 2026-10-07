"""Fetch uv's GitHub releases and split their notes into changelog entries.

Each release body has "## Release Notes" with sections ("### Bug fixes", ...) of
bullets that end in PR links:
    - Handle cycles in `uv pip tree` ([#8689](https://github.com/astral-sh/uv/pull/8689))
One bullet = one changelog_entries row (text without the links + the PR numbers).
That gives the Investigator "which release shipped PR #P" and short, clean fix
descriptions to search. Releases before 0.1.9 (Feb 2024, 10 releases) have no notes.

~4 requests: all pages are re-fetched every run (cheap) and upserted by URL. The
"assets" list (download links, ~9 MB per page) is dropped before storing.

    python -m ingestion.releases
"""
import re
import time

from psycopg.types.json import Jsonb

from ingestion.db import get_db_connection
from ingestion.github_client import get

REPO = "astral-sh/uv"
START_URL = f"https://api.github.com/repos/{REPO}/releases?per_page=100"

SAVE_RAW = """
INSERT INTO raw_responses (repo, url, kind, http_status, body)
VALUES (%s, %s, 'releases_page', %s, %s)
ON CONFLICT (url) DO UPDATE SET
    http_status = EXCLUDED.http_status,
    body        = EXCLUDED.body,
    fetched_at  = now()
"""
UPSERT_RELEASE = """
INSERT INTO releases (repo, tag, published_at, prerelease, body)
VALUES (%s, %s, %s, %s, %s)
ON CONFLICT (repo, tag) DO UPDATE SET
    published_at = EXCLUDED.published_at,
    prerelease   = EXCLUDED.prerelease,
    body         = EXCLUDED.body
"""
DELETE_ENTRIES = "DELETE FROM changelog_entries WHERE repo = %s AND tag = %s"
INSERT_ENTRY = """
INSERT INTO changelog_entries (repo, tag, position, section, text, pr_numbers)
VALUES (%s, %s, %s, %s, %s, %s)
"""

PR_URL = re.compile(r"github\.com/astral-sh/uv/pull/(\d+)")
# A markdown link to a uv PR, e.g. [#8689](https://github.com/astral-sh/uv/pull/8689)
PR_LINK = re.compile(r"\[[^\]]*\]\(https://github\.com/astral-sh/uv/pull/\d+\)")


def parse_notes(body):
    """[(section, text, [pr numbers])] for the bullets under '## Release Notes'."""
    entries, section, in_notes, current = [], None, False, None

    def close():
        if current:
            text = current["text"]
            prs = sorted({int(n) for n in PR_URL.findall(text)})
            text = PR_LINK.sub("", text)
            text = re.sub(r"\(\s*(?:,\s*)*\)", "", text)     # the "( , )" left behind
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                entries.append((current["section"], text, prs))

    for line in (body or "").splitlines():
        if line.startswith("## "):
            close(); current = None
            in_notes = line.strip() == "## Release Notes"
            continue
        if not in_notes:
            continue
        if line.startswith("### "):
            close(); current = None
            section = line[4:].strip()
        elif line.startswith("- "):
            close()
            current = {"section": section, "text": line[2:]}
        elif current and line.startswith("  ") and line.strip():
            current["text"] += " " + line.strip()           # a bullet wrapped onto more lines
        else:
            close(); current = None
    close()
    return entries


def fetch_releases():
    with get_db_connection() as conn:
        url, pages = START_URL, 0
        while url:
            status, body, next_url = get(url)
            if status == 200:
                for r in body:
                    r.pop("assets", None)          # download links: big and not needed
            conn.execute(SAVE_RAW, (REPO, url, status, Jsonb(body)))
            conn.commit()
            if status != 200:
                print(f"Stopped: HTTP {status} on {url}")
                return False
            pages += 1
            print(f"releases page {pages}: {len(body)} releases")
            url = next_url
            time.sleep(0.3)
        return True


def load_releases():
    with get_db_connection() as conn:
        pages = conn.execute(
            "SELECT body FROM raw_responses "
            "WHERE repo = %s AND kind = 'releases_page' AND http_status = 200 ORDER BY id",
            (REPO,),
        ).fetchall()
        n_rel = n_entries = 0
        for (items,) in pages:
            for r in items:
                if r.get("draft") or not r.get("published_at"):
                    continue
                tag = r["tag_name"]
                conn.execute(UPSERT_RELEASE, (REPO, tag, r["published_at"], r["prerelease"], r["body"]))
                conn.execute(DELETE_ENTRIES, (REPO, tag))
                for pos, (section, text, prs) in enumerate(parse_notes(r["body"])):
                    conn.execute(INSERT_ENTRY, (REPO, tag, pos, section, text, prs))
                    n_entries += 1
                n_rel += 1
        conn.commit()
        print(f"{n_rel} releases, {n_entries} changelog entries")


if __name__ == "__main__":
    if fetch_releases():
        load_releases()
