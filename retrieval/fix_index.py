"""Search over merged PRs, for the Investigator ("is this already fixed?").

One vector per merged PR per text variant (table fix_embeddings):
  title_body       PR title + cleaned body (first 1,500 chars): what the developer wrote
  title_changelog  PR title + its release-note bullet(s), or the title alone if it never
                   made the notes: short, user-facing wording ("Handle cycles in `uv pip tree`")
Issue reporters describe symptoms; release notes are written for users, so title_changelog
may match better. Measured, not assumed: python -m evals.score_fix_retrieval.

Bot PRs (dependency bumps) are skipped. Search returns PRs merged strictly before the
cutoff (the triaged issue's created_at), released or not: a fix merged on main but not
yet released is still a useful answer ("fixed on main, coming in the next release").

Embed (resumable per PR and variant; title_changelog is fast, title_body takes longer):
    python -m retrieval.fix_index
"""
from pgvector.psycopg import register_vector

from ingestion.db import get_db_connection
from retrieval.embed import MODEL_NAME
from retrieval.text import issue_text

REPO = "astral-sh/uv"
VARIANTS = ("title_changelog", "title_body")
MAX_BODY_CHARS = 1500
BATCH = 64

# Merged, non-bot PRs that have no vector yet for this variant, with their release-note text.
TODO = """
SELECT p.pr_number, p.title, p.body,
       (SELECT string_agg(e.text, ' ' ORDER BY e.tag, e.position)
          FROM changelog_entries e
         WHERE e.repo = p.repo AND p.pr_number = ANY(e.pr_numbers)) AS notes
FROM pull_requests p
LEFT JOIN fix_embeddings f
  ON f.repo = p.repo AND f.pr_number = p.pr_number AND f.variant = %s
WHERE p.repo = %s AND p.merged_at IS NOT NULL
  AND coalesce(p.user_type, '') <> 'Bot' AND f.pr_number IS NULL
ORDER BY p.pr_number
"""
INSERT = """
INSERT INTO fix_embeddings (repo, pr_number, variant, model, embedding)
VALUES (%s, %s, %s, %s, %s)
"""
NEAREST = """
SELECT f.pr_number
FROM fix_embeddings f
JOIN pull_requests p ON p.repo = f.repo AND p.pr_number = f.pr_number
WHERE f.repo = %(repo)s AND f.variant = %(variant)s
  AND p.merged_at < %(before)s                                  -- existed when the issue was opened
  AND (%(since)s::timestamptz IS NULL OR p.merged_at >= %(since)s)  -- optional recency window
ORDER BY f.embedding <=> %(vec)s
LIMIT %(k)s
"""
# When does the reporter's uv version date from? A fix released before that is already
# in their uv, so it can't be the answer. (Only PRs merged after it are candidates.)
VERSION_DATE = "SELECT published_at FROM releases WHERE repo = %s AND tag = %s"


def pr_text(variant, title, body, notes):
    if variant == "title_changelog":
        return f"{title}\n{notes}" if notes else title
    return issue_text(title, body, MAX_BODY_CHARS)       # same cleaning as issue embeddings


def embed_all(model, variants=VARIANTS):
    with get_db_connection() as conn:
        register_vector(conn)
        for variant in variants:
            rows = conn.execute(TODO, (variant, REPO)).fetchall()
            print(f"{variant}: {len(rows)} PRs to embed")
            for start in range(0, len(rows), BATCH):
                batch = rows[start:start + BATCH]
                texts = [pr_text(variant, t, b, n) for _, t, b, n in batch]
                vectors = model.encode(texts, batch_size=BATCH, normalize_embeddings=True)
                with conn.cursor() as cur:
                    cur.executemany(INSERT, [(REPO, pr, variant, MODEL_NAME, v)
                                             for (pr, *_), v in zip(batch, vectors)])
                conn.commit()
                print(f"  {min(start + BATCH, len(rows))}/{len(rows)}")


def version_date(conn, repo, version):
    """Release date of a uv version like '0.5.3' (None if unknown)."""
    if not version:
        return None
    row = conn.execute(VERSION_DATE, (repo, version)).fetchone()
    return row[0] if row else None


class FixIndex:
    """search(query, before, k, since=None) -> merged PR numbers, best first.
    since: only PRs merged at or after it (a recency window or the reporter's version date)."""

    def __init__(self, conn, repo, variant, model):
        register_vector(conn)
        self.conn, self.repo, self.variant, self.model = conn, repo, variant, model

    def search(self, query, before, k=10, since=None):
        vec = self.model.encode(query, normalize_embeddings=True)
        rows = self.conn.execute(NEAREST, {"repo": self.repo, "variant": self.variant,
                                           "before": before, "since": since, "vec": vec,
                                           "k": k}).fetchall()
        return [r[0] for r in rows]


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    embed_all(SentenceTransformer(MODEL_NAME))
