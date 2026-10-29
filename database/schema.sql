-- =============================================================================
-- database/schema.sql
-- Relational schema for application data (users/auth, catalog cache,
-- explanation logs, evaluation runs). Compatible with both SQLite and
-- PostgreSQL (avoids Postgres-only syntax); see database/db.py for the
-- SQLAlchemy engine that can point at either.
-- =============================================================================

CREATE TABLE IF NOT EXISTS app_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(64) UNIQUE NOT NULL,
    hashed_password VARCHAR(256) NOT NULL,
    role VARCHAR(32) NOT NULL DEFAULT 'viewer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS products_cache (
    product_id VARCHAR(32) PRIMARY KEY,
    category VARCHAR(64),
    brand VARCHAR(64),
    price REAL,
    avg_rating REAL,
    num_reviews INTEGER
);

CREATE TABLE IF NOT EXISTS ranking_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id VARCHAR(32) NOT NULL,
    query TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS explanation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ranking_request_id INTEGER REFERENCES ranking_requests(id),
    product_id VARCHAR(32) NOT NULL,
    rank_score REAL NOT NULL,
    explanation_text TEXT NOT NULL,
    confidence_score REAL NOT NULL,
    ecs_score REAL,
    passed_verification BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name VARCHAR(128) NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Migration marker table so future schema changes can be tracked simply.
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
INSERT INTO schema_migrations (version) SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = 1);
