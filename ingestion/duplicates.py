"""Build duplicate_links (the duplicate-eval answer key) from the stored timelines.

For every issue labelled duplicate or closed as duplicate, read maintainer comments
(MEMBER / OWNER / COLLABORATOR) and collect links to EARLIER issues in our table.
Tiers, per duplicate:
  explicit    - a maintainer comment says "duplicate"/"dupe" and links the issue(s)
  single_link - no such wording, but exactly one earlier issue is linked
  multi_link  - no such wording, two or more earlier issues linked
Duplicates with no usable link are left out. The table is rebuilt from scratch on
every run (one transaction). Run from the project root:  python -m ingestion.duplicates
"""
import re
from collections import Counter

from ingestion.db import get_db_connection

REPO = "astral-sh/uv"
MAINTAINER = {"MEMBER", "OWNER", "COLLABORATOR"}
LINK = re.compile(
    r"github\.com/astral-sh/uv/issues/(\d+)"     # full URL to an issue
    r"|astral-sh/uv#(\d+)"                        # cross-repo shorthand
    r"|(?<![\w/&])#(\d+)\b"                      # plain #N
)
DUP_WORDING = re.compile(r"duplicat|\bdupe", re.I)

DUPES = """
SELECT DISTINCT i.issue_number
FROM issues i
LEFT JOIN issue_labels l ON l.repo = i.repo AND l.issue_number = i.issue_number
WHERE i.repo = %s AND (l.label_name = 'duplicate' OR i.state_reason = 'duplicate')
ORDER BY i.issue_number
"""
TIMELINE_PAGES = """
SELECT body FROM raw_responses
WHERE repo = %s AND kind = 'timeline' AND http_status = 200
  AND url LIKE %s
ORDER BY url
"""
INSERT = """
INSERT INTO duplicate_links (repo, issue_number, original_number, tier, evidence_url)
VALUES (%s, %s, %s, %s, %s)
"""


def linked_issues(text):
    return {int(next(g for g in m.groups() if g)) for m in LINK.finditer(text or "")}


def find_originals(number, events, created):
    """Return (tier, {original_number: evidence_url}) or (None, {}) if nothing usable."""
    explicit, other = {}, {}
    for e in events:
        if e.get("event") != "commented" or e.get("author_association") not in MAINTAINER:
            continue
        body = e.get("body") or ""
        for n in linked_issues(body):
            if n == number or n not in created or created[n] >= created[number]:
                continue                    # self-link, a PR/other repo, or a later issue
            target = explicit if DUP_WORDING.search(body) else other
            target.setdefault(n, e.get("html_url"))
    if explicit:
        return "explicit", explicit
    if len(other) == 1:
        return "single_link", other
    if other:
        return "multi_link", other
    return None, {}


def build():
    with get_db_connection() as conn:
        created = dict(conn.execute(
            "SELECT issue_number, created_at FROM issues WHERE repo = %s", (REPO,)
        ).fetchall())
        dupes = [r[0] for r in conn.execute(DUPES, (REPO,)).fetchall()]

        conn.execute("DELETE FROM duplicate_links WHERE repo = %s", (REPO,))
        tiers, links = Counter(), 0
        for number in dupes:
            events = []
            for (page,) in conn.execute(TIMELINE_PAGES, (REPO, f"%/issues/{number}/timeline%")).fetchall():
                events += page
            tier, originals = find_originals(number, events, created)
            tiers[tier or "no usable link"] += 1
            for original, url in originals.items():
                conn.execute(INSERT, (REPO, number, original, tier, url))
                links += 1
        conn.commit()

    print(f"{len(dupes)} duplicates checked, {links} links saved")
    for tier, n in tiers.most_common():
        print(f"{n:5d}  {tier}")


if __name__ == "__main__":
    build()
