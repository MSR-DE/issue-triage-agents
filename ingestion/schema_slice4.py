"""Slice 4 tables: releases + changelog entries, pull requests, issue comments, and the
"already fixed?" answer key. Safe to re-run (IF NOT EXISTS / constraint re-created).
The same statements are in schema.sql for a fresh setup.

    python -m ingestion.schema_slice4
"""
from ingestion.db import get_db_connection

SQL = """
-- raw_responses gets two new kinds of page.
ALTER TABLE raw_responses DROP CONSTRAINT IF EXISTS raw_responses_kind_check;
ALTER TABLE raw_responses ADD CONSTRAINT raw_responses_kind_check
    CHECK (kind IN ('issues_page', 'timeline', 'releases_page', 'comments_page'));

-- One row per GitHub release (python -m ingestion.releases).
CREATE TABLE IF NOT EXISTS releases (
    repo         TEXT        NOT NULL,
    tag          TEXT        NOT NULL,   -- e.g. '0.4.29'
    published_at TIMESTAMPTZ NOT NULL,
    prerelease   BOOLEAN     NOT NULL,
    body         TEXT,                   -- release notes (markdown)
    PRIMARY KEY (repo, tag)
);

-- One row per bullet in a release's notes: "Fix X ([#8498](.../pull/8498))".
CREATE TABLE IF NOT EXISTS changelog_entries (
    repo       TEXT      NOT NULL,
    tag        TEXT      NOT NULL,
    position   INTEGER   NOT NULL,       -- order within the release notes
    section    TEXT,                     -- 'Bug fixes', 'Enhancements', ...
    text       TEXT      NOT NULL,       -- the bullet without its PR links
    pr_numbers INTEGER[] NOT NULL,
    PRIMARY KEY (repo, tag, position),
    FOREIGN KEY (repo, tag) REFERENCES releases (repo, tag) ON DELETE CASCADE
);

-- Pull requests, taken from the /issues pages already stored in Phase A (no new fetch).
CREATE TABLE IF NOT EXISTS pull_requests (
    repo       TEXT        NOT NULL,
    pr_number  INTEGER     NOT NULL,
    title      TEXT        NOT NULL,
    body       TEXT,
    user_login TEXT,
    user_type  TEXT,
    state      TEXT        NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    closed_at  TIMESTAMPTZ,
    merged_at  TIMESTAMPTZ,              -- NULL = closed without merging, or still open
    PRIMARY KEY (repo, pr_number)
);

-- "Fixes #N" / "Closes #N" in a PR body: the PR says it fixes that issue.
CREATE TABLE IF NOT EXISTS pr_closes (
    repo         TEXT    NOT NULL,
    pr_number    INTEGER NOT NULL,
    issue_number INTEGER NOT NULL,
    PRIMARY KEY (repo, pr_number, issue_number),
    FOREIGN KEY (repo, pr_number)    REFERENCES pull_requests (repo, pr_number) ON DELETE CASCADE,
    FOREIGN KEY (repo, issue_number) REFERENCES issues (repo, issue_number)
);

-- Every comment on an issue or PR (python -m ingestion.comments). No FK to issues:
-- comments on PRs land here too, and the issue table is a snapshot.
CREATE TABLE IF NOT EXISTS issue_comments (
    repo               TEXT        NOT NULL,
    comment_id         BIGINT      NOT NULL,
    issue_number       INTEGER     NOT NULL,
    user_login         TEXT,
    user_type          TEXT,
    author_association TEXT,
    created_at         TIMESTAMPTZ NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL,
    body               TEXT,
    html_url           TEXT,
    PRIMARY KEY (repo, comment_id)
);
CREATE INDEX IF NOT EXISTS issue_comments_issue ON issue_comments (repo, issue_number);

-- Answer key for the Investigator (python -m ingestion.fixes): a maintainer comment on
-- the issue says a PR fixed it ("fixed in #P", "#P fixes this").
--   already_fixed - the PR was merged BEFORE the issue was opened (the bot could have known)
--   fixed_later   - merged after (a normal fix; the bot couldn't have known)
CREATE TABLE IF NOT EXISTS fix_links (
    repo         TEXT        NOT NULL,
    issue_number INTEGER     NOT NULL,
    pr_number    INTEGER     NOT NULL,
    tier         TEXT        NOT NULL CHECK (tier IN ('already_fixed', 'fixed_later')),
    released_in  TEXT,                   -- first release published after the PR was merged
    released_at  TIMESTAMPTZ,
    evidence_url TEXT,                   -- the maintainer comment
    PRIMARY KEY (repo, issue_number, pr_number),
    FOREIGN KEY (repo, issue_number) REFERENCES issues (repo, issue_number),
    FOREIGN KEY (repo, pr_number)    REFERENCES pull_requests (repo, pr_number)
);

-- Merged-PR embeddings for the Investigator's search (python -m retrieval.fix_index).
-- Two text variants per PR; which one searches better is measured.
CREATE TABLE IF NOT EXISTS fix_embeddings (
    repo        TEXT        NOT NULL,
    pr_number   INTEGER     NOT NULL,
    variant     TEXT        NOT NULL CHECK (variant IN ('title_body', 'title_changelog')),
    model       TEXT        NOT NULL,
    embedding   vector(384) NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, pr_number, variant),
    FOREIGN KEY (repo, pr_number) REFERENCES pull_requests (repo, pr_number) ON DELETE CASCADE
);

-- Investigator eval: one row per issue per run (evals/run_investigator.py).
CREATE TABLE IF NOT EXISTS inv_eval_results (
    run_id            TEXT        NOT NULL,   -- e.g. 'invdev-20b-v1'
    repo              TEXT        NOT NULL,
    issue_number      INTEGER     NOT NULL,
    split             TEXT        NOT NULL,   -- 'dev' or 'test'
    kind              TEXT        NOT NULL,   -- 'fixed' or 'control'
    model             TEXT        NOT NULL,
    returned          INTEGER[],              -- grounded fixing PRs, best first; NULL if the run failed
    invented          INTEGER[],
    reported_version  TEXT,
    rounds            INTEGER,
    queries           TEXT[],                 -- the agent's own searches (not the seed)
    reads             INTEGER[],              -- PRs it read
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, repo, issue_number),
    FOREIGN KEY (repo, issue_number) REFERENCES issues (repo, issue_number)
);
"""


def main():
    with get_db_connection() as conn:
        conn.execute(SQL)        # no parameters, so psycopg sends all statements at once
    print("slice 4 tables ready")


if __name__ == "__main__":
    main()
