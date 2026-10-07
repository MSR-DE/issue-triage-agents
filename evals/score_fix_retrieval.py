"""Retrieval eval for the Investigator: for each issue that was ALREADY fixed when it was
opened, does search over merged PRs find the fixing PR?

Answer key: fix_links tier 'already_fixed' (python -m ingestion.fixes). Query = the
issue's title + start of body (what the bot sees); candidates = PRs merged before the
issue was opened. hit@k = a maintainer-linked fixing PR is in the top k.
Compares, all paired on the same issues (no LLM tokens):
  text variant  title_changelog vs title_body (retrieval/fix_index.py)
  window        all merged PRs, or only those merged in the last 90 / 30 / 7 days, or
                only those merged after the reporter's uv version came out ("version";
                falls back to all PRs when the issue states no known version), or both
                ("7d+version": last 7 days AND after the reporter's version, since a PR
                merged before their version was released is already in their uv)
Also prints how long before each issue its fix was merged.

    python -m evals.score_fix_retrieval
"""
from datetime import timedelta
from statistics import median

from evals.dataset import REPO
from evals.score_retrieval import QUERY_BODY_CHARS, compare
from ingestion.db import get_db_connection
from retrieval.fix_index import VARIANTS, FixIndex, version_date
from retrieval.fix_lookup import reported_version
from retrieval.text import issue_text

KS = (1, 5, 10)
WINDOWS = (None, 90, 30, 7, "version", "7d+version")

CASES = """
SELECT f.issue_number, array_agg(f.pr_number), i.title, i.body, i.created_at,
       min(i.created_at - p.merged_at) AS gap
FROM fix_links f
JOIN issues i ON i.repo = f.repo AND i.issue_number = f.issue_number
JOIN pull_requests p ON p.repo = f.repo AND p.pr_number = f.pr_number
WHERE f.repo = %s AND f.tier = 'already_fixed'
GROUP BY f.issue_number, i.title, i.body, i.created_at
ORDER BY f.issue_number
"""


def window_start(conn, window, body, created_at):
    if window is None:
        return None
    if window == "version":
        return version_date(conn, REPO, reported_version(body))
    if window == "7d+version":
        released = version_date(conn, REPO, reported_version(body))
        week = created_at - timedelta(days=7)
        return max(week, released) if released else week
    return created_at - timedelta(days=window)


def evaluate(conn, index, cases, window):
    hits, found5 = {k: 0 for k in KS}, set()
    for number, prs, title, body, created_at, _ in cases:
        since = window_start(conn, window, body, created_at)
        results = index.search(issue_text(title, body, QUERY_BODY_CHARS), before=created_at,
                               k=max(KS), since=since)
        for k in KS:
            hits[k] += bool(set(results[:k]) & set(prs))
        if set(results[:5]) & set(prs):
            found5.add(number)
    return hits, found5


def main(model=None):
    if model is None:
        from sentence_transformers import SentenceTransformer
        from retrieval.embed import MODEL_NAME
        model = SentenceTransformer(MODEL_NAME)
    with get_db_connection() as conn:
        cases = conn.execute(CASES, (REPO,)).fetchall()
        n = len(cases)
        gaps = sorted(g.total_seconds() / 86400 for *_, g in cases)
        with_version = sum(1 for c in cases if version_date(conn, REPO, reported_version(c[3])))
        print(f"{n} already-fixed issues; fix merged before the issue by: median {median(gaps):.1f} days, "
              f"<=7 days {sum(g <= 7 for g in gaps)}, <=30 days {sum(g <= 30 for g in gaps)}, "
              f"max {gaps[-1]:.0f} days")
        print(f"issues stating a known uv version: {with_version}/{n}\n")
        print(f"{'variant':16} {'window':8}" + "".join(f"  recall@{k:<3}" for k in KS))
        found = {}
        for variant in VARIANTS:
            index = FixIndex(conn, REPO, variant, model)
            for window in WINDOWS:
                hits, found[variant, window] = evaluate(conn, index, cases, window)
                label = "all" if window is None else (window if isinstance(window, str) else f"{window}d")
                print(f"{variant:16} {label:8}" + "".join(f"  {hits[k] / n:9.1%}" for k in KS))
        print("\n== paired, top 5 ==")
        compare("title_changelog all", found["title_changelog", None],
                "title_body all", found["title_body", None])
        for variant in VARIANTS:
            for window in WINDOWS[1:]:
                compare(f"{variant} all", found[variant, None], f"{variant} {window}", found[variant, window])
        print("\n== paired, top 5: does adding the version filter help the 7-day window? ==")
        for variant in VARIANTS:
            compare(f"{variant} 7d", found[variant, 7], f"{variant} 7d+version", found[variant, "7d+version"])


if __name__ == "__main__":
    main()
