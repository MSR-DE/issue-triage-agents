"""Plain lookups behind the Duplicate Finder's tools (no LLM, no LangChain).

The agent's @tool wrappers in triage/ call these; the duplicate eval (and a later
MCP server) can call them too. Outputs are short on purpose: every tool result
goes back into the model's prompt, and the free tier allows 8K tokens a minute.
"""
from retrieval.text import clean_body

SNIPPET_CHARS = 200     # per search candidate
READ_CHARS = 1000       # for read_issue

ISSUES_BY_NUMBER = """
SELECT issue_number, title, body
FROM issues
WHERE repo = %s AND issue_number = ANY(%s)
"""

ONE_ISSUE = """
SELECT issue_number, title, body, created_at
FROM issues
WHERE repo = %s AND issue_number = %s
"""


def squash(body, limit):
    """Cleaned body on one line (form boilerplate removed), cut to `limit` characters."""
    text = " ".join(clean_body(body).split())
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def search_candidates(conn, index, repo, query, before, k=10):
    """Top-k issues created before `before`, best first: [{number, title, snippet}].

    `before` is the triaged issue's created_at. The cutoff is applied in SQL by the
    index, so nothing newer can come back, including the issue itself.
    """
    numbers = index.search(query, before=before, k=k)
    if not numbers:
        return []
    rows = conn.execute(ISSUES_BY_NUMBER, (repo, numbers)).fetchall()
    found = {n: (title, body) for n, title, body in rows}
    return [
        {"number": n, "title": found[n][0], "snippet": squash(found[n][1], SNIPPET_CHARS)}
        for n in numbers                      # keep the index's ranking
    ]


def read_issue(conn, repo, number, max_chars=READ_CHARS):
    """One issue's title and cleaned body (cut to max_chars), or None if unknown."""
    row = conn.execute(ONE_ISSUE, (repo, number)).fetchone()
    if row is None:
        return None
    n, title, body, created_at = row
    return {"number": n, "title": title, "created_at": created_at,
            "body": squash(body, max_chars)}
