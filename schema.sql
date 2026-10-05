CREATE TABLE IF NOT EXISTS raw_responses (
    id          BIGSERIAL PRIMARY KEY,
    repo        TEXT NOT NULL,
    url         TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL CHECK (kind IN ('issues_page', 'timeline')),
    http_status INTEGER NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    body        JSONB
);

CREATE TABLE IF NOT EXISTS issues (
    repo            TEXT NOT NULL,
    issue_number    INTEGER NOT NULL,
    github_id       BIGINT NOT NULL,
    title           TEXT NOT NULL,
    body            TEXT,
    state           TEXT NOT NULL,
    state_reason    TEXT,
    user_login      TEXT,
    user_type       TEXT,
    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,
    closed_at       TIMESTAMPTZ,
    raw_response_id BIGINT REFERENCES raw_responses (id),
    PRIMARY KEY (repo, issue_number)
);

CREATE TABLE IF NOT EXISTS issue_labels (
    repo         TEXT NOT NULL,
    issue_number INTEGER NOT NULL,
    label_name   TEXT NOT NULL,
    PRIMARY KEY (repo, issue_number, label_name),
    FOREIGN KEY (repo, issue_number)
        REFERENCES issues (repo, issue_number) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sync_state (
    repo            TEXT PRIMARY KEY,
    phase           TEXT NOT NULL CHECK (phase IN ('A', 'B')),
    next_url        TEXT,
    last_updated_at TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running',
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- One row per (run, issue): what the labeler predicted.
-- Saved after every call, so a stopped run resumes where it left off.
CREATE TABLE IF NOT EXISTS eval_results (
    run_id            TEXT        NOT NULL,   -- e.g. 'dev-20b-v1', 'test-20b-final'
    repo              TEXT        NOT NULL,
    issue_number      INTEGER     NOT NULL,
    split             TEXT        NOT NULL,   -- 'dev' or 'test'
    model             TEXT        NOT NULL,
    gold              TEXT        NOT NULL,   -- maintainer label at run time
    predicted         TEXT,                   -- NULL if the call failed
    raw_output        TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, repo, issue_number),
    FOREIGN KEY (repo, issue_number) REFERENCES issues (repo, issue_number)
);

-- Ground truth for the duplicate eval: which earlier issue(s) each duplicate points to.
-- Rebuilt from timelines by `python -m ingestion.duplicates`.
CREATE TABLE IF NOT EXISTS duplicate_links (
    repo            TEXT    NOT NULL,
    issue_number    INTEGER NOT NULL,   -- the duplicate
    original_number INTEGER NOT NULL,   -- an earlier issue a maintainer pointed to
    tier            TEXT    NOT NULL CHECK (tier IN ('explicit', 'single_link', 'multi_link')),
    evidence_url    TEXT,               -- the maintainer comment with the link
    PRIMARY KEY (repo, issue_number, original_number),
    FOREIGN KEY (repo, issue_number)    REFERENCES issues (repo, issue_number),
    FOREIGN KEY (repo, original_number) REFERENCES issues (repo, issue_number)
);

-- pgvector: needed for issue_embeddings.
CREATE EXTENSION IF NOT EXISTS vector;

-- One embedding per issue, kept in its own table so issues stays the raw original
-- and the model can be swapped later (the model column says which one made the vector).
CREATE TABLE IF NOT EXISTS issue_embeddings (
    repo         TEXT        NOT NULL,
    issue_number INTEGER     NOT NULL,
    model        TEXT        NOT NULL,          -- e.g. 'BAAI/bge-small-en-v1.5'
    embedding    vector(384) NOT NULL,          -- bge-small outputs 384 numbers
    embedded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (repo, issue_number),
    FOREIGN KEY (repo, issue_number)
        REFERENCES issues (repo, issue_number) ON DELETE CASCADE
);

-- Duplicate Finder eval: one row per issue per run (evals/run_dup_finder.py).
CREATE TABLE IF NOT EXISTS dup_eval_results (
    run_id            TEXT        NOT NULL,   -- e.g. 'dupdev-20b-v1'
    repo              TEXT        NOT NULL,
    issue_number      INTEGER     NOT NULL,
    split             TEXT        NOT NULL,   -- 'dev' or 'test'
    kind              TEXT        NOT NULL,   -- 'duplicate' or 'control'
    model             TEXT        NOT NULL,
    returned          INTEGER[],              -- grounded answer, best first; NULL if the run failed
    invented          INTEGER[],              -- numbers the verdict made up (dropped)
    rounds            INTEGER,
    queries           TEXT[],                 -- the agent's own search queries (not the seed)
    reads             INTEGER[],              -- issues it read
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    error             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, repo, issue_number),
    FOREIGN KEY (repo, issue_number) REFERENCES issues (repo, issue_number)
);
