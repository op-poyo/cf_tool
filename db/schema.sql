-- Codeforces analytics tool schema

CREATE TABLE IF NOT EXISTS contests (
    id            INTEGER PRIMARY KEY,   -- CF contest id
    name          TEXT,
    phase         TEXT,                  -- e.g. FINISHED, BEFORE
    type          TEXT,                  -- CF, IOI, ICPC
    start_time    INTEGER,               -- unix seconds
    duration      INTEGER,               -- seconds
    fetched_at    INTEGER NOT NULL       -- when we cached this row
);

CREATE TABLE IF NOT EXISTS problems (
    contest_id     INTEGER NOT NULL,
    problem_index  TEXT NOT NULL,        -- 'A', 'B', ... as CF gives it
    name           TEXT,
    rating         INTEGER,              -- NULL if unrated problem
    fetched_at     INTEGER NOT NULL,
    PRIMARY KEY (contest_id, problem_index)
);

CREATE TABLE IF NOT EXISTS problem_tags (
    contest_id     INTEGER NOT NULL,
    problem_index  TEXT NOT NULL,
    tag            TEXT NOT NULL,
    PRIMARY KEY (contest_id, problem_index, tag),
    FOREIGN KEY (contest_id, problem_index)
        REFERENCES problems (contest_id, problem_index)
);

CREATE TABLE IF NOT EXISTS users (
    handle           TEXT PRIMARY KEY,
    current_rating   INTEGER,
    last_synced_at   INTEGER
);

CREATE TABLE IF NOT EXISTS user_rating_history (
    handle               TEXT NOT NULL,
    contest_id           INTEGER NOT NULL,
    rating_before        INTEGER,
    rating_after         INTEGER,
    rank                 INTEGER,
    rating_update_time   INTEGER,
    PRIMARY KEY (handle, contest_id)
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id  INTEGER PRIMARY KEY, -- CF's own submission id, globally unique
    handle         TEXT NOT NULL,
    contest_id     INTEGER,
    problem_index  TEXT,
    verdict        TEXT,
    creation_time  INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_submissions_handle_problem
    ON submissions (handle, contest_id, problem_index);

CREATE INDEX IF NOT EXISTS idx_submissions_handle_time
    ON submissions (handle, creation_time);

-- Tracks freshness of cached data.
-- key examples: 'global:contest_list', 'global:problemset', 'user:<handle>'
CREATE TABLE IF NOT EXISTS cache_meta (
    key                TEXT PRIMARY KEY,
    last_refreshed_at  INTEGER NOT NULL
);

-- Everything below is local-only, entered by the user -- Codeforces'
-- API never tells you when you started a problem or how it felt, only
-- when you submitted.
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    handle        TEXT NOT NULL,
    started_at    INTEGER NOT NULL,
    ended_at      INTEGER,               -- NULL while the session is active
    notes         TEXT
);

CREATE INDEX IF NOT EXISTS idx_sessions_handle
    ON sessions (handle, started_at);

CREATE TABLE IF NOT EXISTS session_attempts (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id        INTEGER NOT NULL,
    contest_id        INTEGER NOT NULL,
    problem_index     TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'in_progress',  -- in_progress, solved, gave_up
    timer_state       TEXT NOT NULL DEFAULT 'stopped',      -- running, paused, stopped
    active_seconds    INTEGER NOT NULL DEFAULT 0,           -- accumulated elapsed time; excludes paused spans
    last_resumed_at   INTEGER,                              -- unix seconds; set only while timer_state = 'running'
    felt_difficulty   TEXT,                                 -- free-form, e.g. 'easier than rated' / 'as expected' / 'harder'
    notes             TEXT,
    created_at        INTEGER NOT NULL,
    solved_at         INTEGER,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

CREATE INDEX IF NOT EXISTS idx_session_attempts_session
    ON session_attempts (session_id);
