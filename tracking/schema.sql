PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    source_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    url TEXT,
    local_path TEXT,
    accessibility TEXT,
    license TEXT,
    confidence TEXT,
    last_checked TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS claims (
    claim_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    claim TEXT NOT NULL,
    evidence_class TEXT NOT NULL,
    confidence TEXT NOT NULL,
    source_id TEXT,
    implication TEXT,
    status TEXT NOT NULL DEFAULT 'hypothesis',
    FOREIGN KEY(source_id) REFERENCES sources(source_id)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    statement TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 99,
    evidence_for TEXT,
    evidence_against TEXT,
    next_test TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    decided_at TEXT NOT NULL,
    title TEXT NOT NULL,
    decision TEXT NOT NULL,
    rationale TEXT,
    source_ids TEXT,
    revisit_trigger TEXT
);

CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    status TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 99,
    phase TEXT NOT NULL,
    depends_on TEXT,
    due_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    status TEXT NOT NULL,
    track TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 99,
    parent_id TEXT,
    owner TEXT,
    manifest_path TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(parent_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    status TEXT NOT NULL,
    git_sha TEXT,
    fold_map TEXT,
    config_path TEXT,
    artifact_dir TEXT,
    controls_json TEXT,
    notes TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    split TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    unit TEXT,
    fold TEXT NOT NULL DEFAULT '',
    well_id TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(run_id, split, name, fold, well_id),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS well_metrics (
    run_id TEXT NOT NULL,
    well_id TEXT NOT NULL,
    split TEXT NOT NULL,
    rows_scored INTEGER,
    rmse REAL,
    mean_error REAL,
    sse REAL,
    regime TEXT,
    uncertainty REAL,
    PRIMARY KEY(run_id, well_id, split),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS submissions (
    submission_id TEXT PRIMARY KEY,
    run_id TEXT,
    kaggle_ref TEXT,
    submitted_at TEXT NOT NULL,
    public_score REAL,
    private_score REAL,
    description TEXT,
    selected INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT,
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    sha256 TEXT,
    bytes INTEGER,
    created_at TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sync_events (
    sync_id INTEGER PRIMARY KEY AUTOINCREMENT,
    synced_at TEXT NOT NULL,
    source_path TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_lookup ON metrics(split, name, value);
CREATE INDEX IF NOT EXISTS idx_well_metrics_run ON well_metrics(run_id, split);
CREATE INDEX IF NOT EXISTS idx_submissions_date ON submissions(submitted_at);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, priority);

CREATE VIEW IF NOT EXISTS v_run_cv AS
SELECT r.run_id, r.experiment_id, r.status,
       MIN(CASE WHEN m.split IN ('cv','oof') AND m.name='rmse' THEN m.value END) AS cv_rmse,
       MAX(CASE WHEN m.split IN ('cv','oof') AND m.name='worst_5pct_sse_share' THEN m.value END) AS worst_5pct_sse_share,
       MAX(CASE WHEN m.split IN ('cv','oof') AND m.name='worst_10pct_sse_share' THEN m.value END) AS worst_10pct_sse_share,
       MAX(CASE WHEN m.split IN ('cv','oof') AND m.name='median_well_rmse' THEN m.value END) AS median_well_rmse,
       MAX(CASE WHEN m.split IN ('cv','oof') AND m.name='p90_well_rmse' THEN m.value END) AS p90_well_rmse
FROM runs r
LEFT JOIN metrics m ON m.run_id=r.run_id
GROUP BY r.run_id, r.experiment_id, r.status;

CREATE VIEW IF NOT EXISTS v_experiment_summary AS
SELECT e.experiment_id, e.title, e.hypothesis, e.status, e.track, e.priority,
       COUNT(DISTINCT r.run_id) AS run_count,
       MIN(v.cv_rmse) AS best_cv_rmse,
       MAX(r.updated_at) AS last_run_at
FROM experiments e
LEFT JOIN runs r ON r.experiment_id=e.experiment_id
LEFT JOIN v_run_cv v ON v.run_id=r.run_id
GROUP BY e.experiment_id;
