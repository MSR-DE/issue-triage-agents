"""Vector search: find the stored issue embeddings closest to a new issue's embedding.

Same interface as BM25Index (search(query, before, k) -> issue numbers), so the eval
can score both the same way. The query is embedded live with the same model and the
same text recipe as embed.py, exactly as the bot would do for a brand-new issue.
"""
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from retrieval.embed import MODEL_NAME
from retrieval.text import issue_text  # noqa: F401  (callers build the query with it)

# <=> is pgvector's cosine DISTANCE (0 = same direction). Smallest distance = most similar.
# No index on the column, so Postgres compares against every row: exact results.
NEAREST = """
SELECT e.issue_number
FROM issue_embeddings e
JOIN issues i ON i.repo = e.repo AND i.issue_number = e.issue_number
WHERE e.repo = %s
  AND i.created_at < %s          -- no peeking at the future
ORDER BY e.embedding <=> %s
LIMIT %s
"""


class VectorIndex:
    def __init__(self, conn, repo):
        register_vector(conn)            # lets psycopg send a numpy array as a vector
        self.conn = conn                 # kept open: every search is one SQL query
        self.repo = repo
        self.model = SentenceTransformer(MODEL_NAME)

    def search(self, query, before, k=10):
        vec = self.model.encode(query, normalize_embeddings=True)   # same settings as embed.py
        rows = self.conn.execute(NEAREST, (self.repo, before, vec, k)).fetchall()
        return [r[0] for r in rows]