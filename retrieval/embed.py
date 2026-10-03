"""Embed every issue once and store the vectors in Postgres (pgvector).

What an embedding is: the model reads a text and outputs 384 numbers. Texts that MEAN
similar things get vectors pointing in similar directions, even when the words differ
("hangs behind a proxy" vs "stalls when HTTPS_PROXY is set"). Searching later = find
the stored vectors closest to a new issue's vector.

Resumable: only issues without a row in issue_embeddings get embedded, and every batch
is committed, so Ctrl+C and re-run continues where it stopped.
Run: python -m retrieval.embed
"""
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from evals.dataset import REPO
from ingestion.db import get_db_connection
from retrieval.text import issue_text

MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_BODY_CHARS = 1500   # the model reads ~512 tokens (~2,000 chars), so title + 1,500 chars fits
BATCH = 64              # issues per model call: batching is much faster than one at a time

# LEFT JOIN + "IS NULL" = issues that have no embedding row yet. This is what makes it resumable.
TODO = """
SELECT i.issue_number, i.title, i.body
FROM issues i
LEFT JOIN issue_embeddings e
  ON e.repo = i.repo AND e.issue_number = i.issue_number
WHERE i.repo = %s AND i.user_type <> 'Bot' AND e.issue_number IS NULL
ORDER BY i.issue_number
"""

INSERT = """
INSERT INTO issue_embeddings (repo, issue_number, model, embedding)
VALUES (%s, %s, %s, %s)
"""


def main():
    model = SentenceTransformer(MODEL_NAME)
    with get_db_connection() as conn:
        register_vector(conn)    # teaches psycopg to send numpy arrays as pgvector values
        rows = conn.execute(TODO, (REPO,)).fetchall()
        print(f"{len(rows)} issues to embed")

        for start in range(0, len(rows), BATCH):
            batch = rows[start:start + BATCH]
            # Same text BM25 searches over: title + cleaned body (boilerplate removed).
            # No "query instruction" prefix: we compare issue to issue, so both sides
            # are the same kind of text.
            texts = [issue_text(title, body, MAX_BODY_CHARS) for _, title, body in batch]

            # normalize_embeddings=True scales every vector to length 1. Then cosine
            # similarity is just a dot product, and all distances are on the same scale.
            vectors = model.encode(texts, batch_size=BATCH, normalize_embeddings=True)

            with conn.cursor() as cur:
                # executemany = run INSERT once per (issue, vector) pair
                cur.executemany(
                    INSERT,
                    [(REPO, number, MODEL_NAME, vec) for (number, _, _), vec in zip(batch, vectors)],
                )
            conn.commit()        # one commit per batch: a crash loses at most 64 issues of work
            print(f"{min(start + BATCH, len(rows))}/{len(rows)}")


if __name__ == "__main__":
    main()