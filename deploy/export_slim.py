"""Copy the slim database the live demo needs to a hosted Postgres (Neon free tier, 0.5 GB).

    python -m deploy.export_slim            # source: DATABASE_URL, target: DEMO_DATABASE_URL (.env)

Copied: issues (+ labels, embeddings), merged-PR data for the Investigator (pull_requests,
fix_embeddings for the variant it uses), releases + changelog entries. Not copied: raw API
pages, comments, answer keys and eval results (hundreds of MB the demo never reads).
Re-runnable: the target tables are emptied first. Tables are created from schema.sql, so
the demo database has the same shape as the local one (unused tables just stay empty).
"""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from triage.investigator import INDEX_VARIANT

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
REPO = "astral-sh/uv"

# (table, columns, row filter) in foreign-key order.
TABLES = [
    ("issues", "repo, issue_number, github_id, title, body, state, state_reason, user_login, "
               "user_type, created_at, updated_at, closed_at", ""),
    ("issue_labels", "repo, issue_number, label_name", ""),
    ("issue_embeddings", "repo, issue_number, model, embedding, embedded_at", ""),
    ("pull_requests", "repo, pr_number, title, body, user_login, user_type, state, created_at, "
                      "closed_at, merged_at", ""),
    ("releases", "repo, tag, published_at, prerelease, body", ""),
    ("changelog_entries", "repo, tag, position, section, text, pr_numbers", ""),
    ("fix_embeddings", "repo, pr_number, variant, model, embedding, embedded_at",
     f"AND variant = '{INDEX_VARIANT}'"),
]


def copy_table(src, dst, table, cols, extra):
    query = f"COPY (SELECT {cols} FROM {table} WHERE repo = '{REPO}' {extra}) TO STDOUT"
    with src.cursor().copy(query) as out, dst.cursor().copy(f"COPY {table} ({cols}) FROM STDIN") as inp:
        for chunk in out:            # streamed: never holds a whole table in memory
            inp.write(chunk)
    n_src = src.execute(f"SELECT count(*) FROM {table} WHERE repo = %s {extra}", (REPO,)).fetchone()[0]
    n_dst = dst.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    if n_src != n_dst:
        raise RuntimeError(f"{table}: {n_src} rows in source, {n_dst} copied")
    print(f"{table}: {n_dst} rows")


def main():
    target = os.getenv("DEMO_DATABASE_URL")
    if not target:
        raise SystemExit("DEMO_DATABASE_URL missing from .env (the Neon connection string)")
    with psycopg.connect(os.environ["DATABASE_URL"]) as src, psycopg.connect(target) as dst:
        dst.execute((ROOT / "schema.sql").read_text(encoding="utf-8"))
        dst.execute("TRUNCATE " + ", ".join(t for t, _, _ in reversed(TABLES)) + " CASCADE")
        for table, cols, extra in TABLES:
            copy_table(src, dst, table, cols, extra)
        dst.commit()
        dst.execute("ANALYZE")
        size = dst.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
    print(f"done; demo database size: {size} (Neon free tier: 0.5 GB)")


if __name__ == "__main__":
    main()
