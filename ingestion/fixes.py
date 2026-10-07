"""Build fix_links (the Investigator's answer key) from maintainer comments.

A maintainer comment (MEMBER / OWNER / COLLABORATOR, not a bot) on an issue that says
a merged uv PR fixed it, in one of two shapes (quotes and code blocks removed first,
since those are usually the reporter's words or logs):
    "fixed in #P" / "fixed by #P" / "resolved via <PR url>" / "fixed this in #P"
    "#P fixes this" / "#P should fix it" / "#P resolved this"
Tier from timing:
    already_fixed - P was merged BEFORE the issue was opened: the fix already existed,
                    so a bot reading the issue could have said "upgrade" (or "fixed on
                    main, not released yet" if no release had shipped P by then)
    fixed_later   - merged after: an ordinary fix, invisible at report time
released_in = the first release published after P was merged (uv releases from main,
so that is the release that shipped it, even for PRs the notes don't list). Rebuilt from scratch each run.
Comments that only name a version ("fixed in 0.5.22") are counted, not stored: they
have no PR to score against.

    python -m ingestion.fixes              # rebuild
    python -m ingestion.fixes show 15      # spot-check 15 random already_fixed links
"""
import re
from collections import Counter

from ingestion.db import get_db_connection

REPO = "astral-sh/uv"

# Every way a comment points at a uv PR, rewritten to "#N" before matching.
MD_LINK = re.compile(r"\[[^\]]*\]\((https?://github\.com/astral-sh/uv/(?:pull|issues)/\d+)[^)]*\)")
URL_REF = re.compile(r"https?://github\.com/astral-sh/uv/(?:pull|issues)/(\d+)\S*")
SHORT_REF = re.compile(r"\bastral-sh/uv#(\d+)")
CODE_BLOCK = re.compile(r"```.*?```", re.S)
FIX_VERB = r"(?:fix(?:ed|es)?|resolved?|resolves|addressed|address(?:es)?|solved?|solves)"
AFTER = re.compile(                       # "fixed in #P", "fixed already in #P and #Q"
    r"\b" + FIX_VERB + r"(?:\s+(?:this|it|that))?(?:\s+(?:already|now|upstream|on main))?"
    r"\s+(?:in|by|via|with)\s+(?:(?:PRs?|pull requests?)\s+)?(#\d+(?:\s*(?:,|and|&)\s*#\d+)*)",
    re.I)
NUM = re.compile(r"#(\d+)")
BEFORE = re.compile(                      # "#P fixes this", "#P should fix it"
    r"(?<![\w&])#(\d+)\s+(?:should\s+(?:have\s+)?|will\s+|has\s+|also\s+)?" + FIX_VERB + r"\b",
    re.I)
# Hedges that mean "NOT (really) fixed". The hand check of all 34 already_fixed links
# (7 Oct 2026) found 7 wrong, every one hedged: "I thought we fixed this in #P", "had
# hoped to have fixed", "purportedly fixed by", "I presume?", "partially fixed",
# "careful not to break the things we fixed in #P". Such comments are skipped.
HEDGE = re.compile(r"\b(?:thought|hoped|purportedly|presume|partially|not to break)\b", re.I)
VERSION_ONLY = re.compile(r"\b" + FIX_VERB + r"(?:\s+(?:this|it))?\s+in\s+(?:uv\s+)?v?\d+\.\d+\.\d+", re.I)

COMMENTS = """
SELECT c.issue_number, c.body, c.html_url, c.created_at, i.created_at
FROM issue_comments c
JOIN issues i ON i.repo = c.repo AND i.issue_number = c.issue_number
WHERE c.repo = %s
  AND c.author_association IN ('MEMBER', 'OWNER', 'COLLABORATOR')
  AND coalesce(c.user_type, '') <> 'Bot'
  AND i.user_type <> 'Bot'
ORDER BY c.issue_number, c.created_at
"""
MERGED = "SELECT pr_number, merged_at FROM pull_requests WHERE repo = %s AND merged_at IS NOT NULL"
FIRST_RELEASE = """
SELECT r.tag, r.published_at FROM releases r
WHERE r.repo = %s AND NOT r.prerelease AND r.published_at > %s
ORDER BY r.published_at LIMIT 1
"""
INSERT = """
INSERT INTO fix_links (repo, issue_number, pr_number, tier, released_in, released_at, evidence_url)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (repo, issue_number, pr_number) DO NOTHING
"""


def normalize(body):
    """Comment text with quotes and code removed and every PR reference as #N."""
    body = CODE_BLOCK.sub(" ", body or "")
    body = "\n".join(l for l in body.splitlines() if not l.lstrip().startswith(">"))
    body = MD_LINK.sub(r"\1", body)
    body = URL_REF.sub(r"#\1", body)
    return SHORT_REF.sub(r"#\1", body)


def fixing_prs(body):
    """PR numbers a comment says fixed the issue (not yet checked against merged PRs)."""
    text = normalize(body)
    after = {int(n) for refs in AFTER.findall(text) for n in NUM.findall(refs)}
    return after | {int(n) for n in BEFORE.findall(text)}


def build():
    with get_db_connection() as conn:
        merged = dict(conn.execute(MERGED, (REPO,)).fetchall())
        rows = conn.execute(COMMENTS, (REPO,)).fetchall()

        conn.execute("DELETE FROM fix_links WHERE repo = %s", (REPO,))
        tiers, version_only, not_merged = Counter(), set(), 0
        for issue, body, url, _, opened in rows:
            prs = set() if HEDGE.search(normalize(body)) else fixing_prs(body) - {issue}
            if not prs and VERSION_ONLY.search(normalize(body)):
                version_only.add(issue)
            for pr in prs:
                if pr not in merged:          # an issue number, or a PR never merged
                    not_merged += 1
                    continue
                tier = "already_fixed" if merged[pr] < opened else "fixed_later"
                tag, at = conn.execute(FIRST_RELEASE, (REPO, merged[pr])).fetchone() or (None, None)
                cur = conn.execute(INSERT, (REPO, issue, pr, tier, tag, at, url))
                tiers[tier] += cur.rowcount
        conn.commit()

        print(f"{len(rows)} maintainer comments read; {not_merged} references skipped (not a merged PR)")
        print(f"fix links: {dict(tiers)}")
        for tier in ("already_fixed", "fixed_later"):
            n = conn.execute("SELECT count(DISTINCT issue_number) FROM fix_links "
                             "WHERE repo = %s AND tier = %s", (REPO, tier)).fetchone()[0]
            print(f"  {tier}: {n} issues")
        released_before = conn.execute(
            "SELECT count(DISTINCT f.issue_number) FROM fix_links f JOIN issues i "
            "ON i.repo = f.repo AND i.issue_number = f.issue_number "
            "WHERE f.repo = %s AND f.tier = 'already_fixed' AND f.released_at < i.created_at",
            (REPO,)).fetchone()[0]
        print(f"  already_fixed and already released when reported: {released_before} issues "
              f"(the rest were fixed on main, not yet released)")
        print(f"version-only 'fixed in X.Y.Z' comments (not stored): {len(version_only)} issues")


SAMPLE = """
SELECT f.issue_number, i.title, i.created_at::date, f.pr_number, p.title, p.merged_at::date,
       f.released_in, c.body
FROM fix_links f
JOIN issues i ON i.repo = f.repo AND i.issue_number = f.issue_number
JOIN pull_requests p ON p.repo = f.repo AND p.pr_number = f.pr_number
JOIN issue_comments c ON c.repo = f.repo AND c.html_url = f.evidence_url
WHERE f.repo = %s AND f.tier = 'already_fixed'
ORDER BY md5(f.issue_number::text || f.pr_number::text)   -- fixed pseudo-random order
LIMIT %s
"""


def show(n):
    """Print already_fixed links for a hand check: is the PR really the fix for the issue?"""
    with get_db_connection() as conn:
        for row in conn.execute(SAMPLE, (REPO, n)).fetchall():
            issue, ititle, opened, pr, ptitle, merged, rel, comment = row
            print(f"#{issue} ({opened}) {ititle}")
            print(f"  PR #{pr} merged {merged}, released in {rel or '-'}: {ptitle}")
            print(f"  comment: {' '.join((comment or '').split())[:300]}\n")


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["show"]:
        show(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    else:
        build()
