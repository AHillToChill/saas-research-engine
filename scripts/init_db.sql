-- scripts/init_db.sql
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lane TEXT NOT NULL,                 -- A | B | C
  source_type TEXT NOT NULL,          -- stackexchange | reddit | forum | reviews | etc.
  site TEXT,                          -- e.g., "stackoverflow"
  query TEXT,                         -- the query used to collect
  collected_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_id INTEGER NOT NULL,
  external_id TEXT NOT NULL,          -- e.g., StackExchange question_id
  url TEXT NOT NULL,
  title TEXT,
  body_text TEXT,                     -- extracted plain text (not full HTML)
  created_at_utc TEXT,
  score INTEGER,
  num_answers INTEGER,
  tags TEXT,                          -- comma-separated
  raw_json TEXT,                      -- store API response JSON for traceability
  ingested_at_utc TEXT NOT NULL,
  UNIQUE(source_id, external_id),
  FOREIGN KEY(source_id) REFERENCES sources(id)
);

CREATE TABLE IF NOT EXISTS pain_units (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  raw_document_id INTEGER NOT NULL,
  lane TEXT NOT NULL,                 -- A | B | C
  actor TEXT,                         -- "dispatch manager", "analyst", "simulation engineer"
  task TEXT,
  friction TEXT,
  consequence TEXT,
  normalized_pain TEXT NOT NULL,      -- "As a ___, I struggle to ___ because ___"
  workaround TEXT,
  severity INTEGER,                   -- 1..5
  frequency INTEGER,                  -- 1..5 (initially heuristic)
  purchase_intent INTEGER,            -- 0..3
  stage TEXT,                         -- onboarding | daily | scaling | edge-case
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY(raw_document_id) REFERENCES raw_documents(id)
);

CREATE TABLE IF NOT EXISTS clusters (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lane TEXT NOT NULL,
  method TEXT NOT NULL,               -- tfidf_kmeans_v1, etc.
  label TEXT,                         -- human-readable name we assign later
  summary TEXT,                       -- short description of cluster
  score REAL,                         -- composite ranking
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS cluster_members (
  cluster_id INTEGER NOT NULL,
  pain_unit_id INTEGER NOT NULL,
  PRIMARY KEY(cluster_id, pain_unit_id),
  FOREIGN KEY(cluster_id) REFERENCES clusters(id),
  FOREIGN KEY(pain_unit_id) REFERENCES pain_units(id)
);
