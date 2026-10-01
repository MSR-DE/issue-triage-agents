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