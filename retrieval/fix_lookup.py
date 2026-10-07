"""Plain lookups behind the Investigator's tools (no LLM, no LangChain).

Every fact about a candidate fix is computed in code relative to the triaged issue:
merged before it was opened (the index guarantees that), released before it or not
(the first release published after the merge), and whether the reporter's uv version
already includes that release. The model judges "is this the same problem?"; dates
and versions are never left to it.
"""
import re

from retrieval.lookup import squash

SNIPPET_CHARS = 200
READ_CHARS = 1000

# "uv 0.4.18" / "uv-0.4.18" / "uv v0.4.18" anywhere in the issue (the bug form asks for `uv --version`).
UV_VERSION = re.compile(r"\buv[ -]v?(\d+\.\d+\.\d+)\b", re.I)

PRS = """
SELECT p.pr_number, p.title, p.body, p.merged_at,
       (SELECT string_agg(e.text, ' ' ORDER BY e.tag, e.position)
          FROM changelog_entries e
         WHERE e.repo = p.repo AND p.pr_number = ANY(e.pr_numbers)) AS notes,
       -- uv releases from main, so a merged PR ships in the first release after its merge
       -- (internal PRs too, which the notes don't list). Only releases that existed then.
       (SELECT r.tag FROM releases r
         WHERE r.repo = p.repo AND NOT r.prerelease
           AND r.published_at > p.merged_at AND r.published_at < %s
         ORDER BY r.published_at LIMIT 1) AS released_in
FROM pull_requests p
WHERE p.repo = %s AND p.pr_number = ANY(%s) AND p.merged_at < %s
"""
LATEST_RELEASE = """
SELECT tag FROM releases
WHERE repo = %s AND NOT prerelease AND published_at < %s
ORDER BY published_at DESC LIMIT 1
"""


def version_tuple(v):
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", v or "")
    return tuple(int(x) for x in m.groups()) if m else None


def reported_version(body):
    """The uv version the reporter quotes, e.g. '0.4.18', or None."""
    m = UV_VERSION.search(body or "")
    return m.group(1) if m else None


def latest_release(conn, repo, before):
    row = conn.execute(LATEST_RELEASE, (repo, before)).fetchone()
    return row[0] if row else None


def fix_status(merged_at, released_in, reported):
    """One line of facts about a candidate fix, as of when the issue was opened."""
    if not released_in:
        return f"merged {merged_at:%Y-%m-%d}, not yet in a release when the issue was opened"
    status = f"released in {released_in}"
    rel, rep = version_tuple(released_in), version_tuple(reported)
    if rel and rep:
        status += (f", newer than the reporter's uv {reported}" if rel > rep
                   else f"; the reporter's uv {reported} already includes it")
    return status


def _rows(conn, repo, numbers, before):
    rows = conn.execute(PRS, (before, repo, list(numbers), before)).fetchall()
    return {r[0]: r for r in rows}


def search_fix_candidates(conn, index, repo, query, before, reported=None, k=8, since=None):
    """Top-k PRs merged before `before` (and at or after `since`, if given), best first:
    [{number, title, text, status}]."""
    numbers = index.search(query, before=before, k=k, since=since)
    found = _rows(conn, repo, numbers, before)
    out = []
    for n in numbers:
        if n not in found:
            continue
        _, title, body, merged_at, notes, released_in = found[n]
        out.append({"number": n, "title": title,
                    "text": squash(notes or body, SNIPPET_CHARS),
                    "status": fix_status(merged_at, released_in, reported)})
    return out


def read_pr(conn, repo, number, before, reported=None, max_chars=READ_CHARS):
    """One merged PR (merged before `before`): title, release note, body, status; or None."""
    row = _rows(conn, repo, [number], before).get(number)
    if row is None:
        return None
    _, title, body, merged_at, notes, released_in = row
    return {"number": number, "title": title, "notes": notes or "",
            "body": squash(body, max_chars),
            "status": fix_status(merged_at, released_in, reported)}
