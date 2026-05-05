--- @module ion7.rag.db.schema
--- @author  ion7 / Ion7 Project Contributors
---
--- Versioned DDL for the two SQLite files ion7-rag uses :
---
---   chunks.db (main)        canonical text + metadata + provenance
---   index.db  (idx)         sqlite-vec dense vectors + FTS5 BM25 +
---                            HyPE question vectors
---
--- Each database carries a `meta(key, value)` table that records its
--- schema version, and (for `index.db`) the embedder-specific
--- dimensions baked into the vec0 column types. The bootstrap path is
--- forward-compatible : every CREATE is `IF NOT EXISTS`, so opening a
--- DB written by an older build adds the missing tables and bumps the
--- stamp. Mismatched runtime `embed_dim` / `binary_dim` on re-open
--- raises rather than silently corrupting the index.

local M = {}

--- Current schema version. Stored in each DB's `meta` table on
--- creation and used by the bootstrap path to detect older DBs that
--- need a forward migration.
M.SCHEMA_VERSION = 2

-- ── DDL : chunks.db ─────────────────────────────────────────────────────

M.CHUNKS_DDL = [[
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS docs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL UNIQUE,
    source_uri  TEXT,
    title       TEXT,
    format      TEXT NOT NULL,
    ingested_at INTEGER NOT NULL,
    meta_json   TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_pk          INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    section         TEXT,
    char_start      INTEGER NOT NULL,
    char_end        INTEGER NOT NULL,
    n_tokens        INTEGER,
    raw_text        TEXT NOT NULL,
    contextual_text TEXT,
    meta_json       TEXT
);

CREATE INDEX IF NOT EXISTS chunks_doc_pk ON chunks(doc_pk);
]]

-- ── DDL : index.db (attached as `idx`) ──────────────────────────────────

--- Build the DDL for `index.db`. vec0 column types are dimension-bound
--- at creation, so the binary and full-precision dims are interpolated
--- from runtime `opts`. The `idx.meta` table records both so a later
--- open can validate them.
---
--- Tables :
---   `idx.meta`         schema version + embed_dim / binary_dim stamps.
---   `idx.chunks_vec`   chunk-level vectors. PK = `chunks.id`. Two tiers :
---                      `embedding_bin` (binary-quantised, MRL-truncated
---                      shortlist) and `embedding_full` (fp32 rerank).
---   `idx.chunks_fts`   FTS5 BM25 index on chunk text.
---   `idx.hype_vec`     HyPE-question vectors. Uses vec0 auxiliary
---                      columns (`+chunk_id`, `+question`) so a single
---                      KNN MATCH returns the parent chunk_id and the
---                      raw question text without a join.
---
--- @param  opts table { embed_dim = integer, binary_dim = integer }
--- @return string  Multi-statement SQL ready for `conn:exec`.
--- @raise When either dimension is missing.
function M.index_ddl(opts)
    local embed_dim  = assert(opts.embed_dim,
        "schema.index_ddl : embed_dim required")
    local binary_dim = assert(opts.binary_dim,
        "schema.index_ddl : binary_dim required")

    return string.format([[
CREATE TABLE IF NOT EXISTS idx.meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS idx.chunks_vec USING vec0(
    chunk_id       INTEGER PRIMARY KEY,
    embedding_bin  BIT[%d],
    embedding_full FLOAT[%d]
);

CREATE VIRTUAL TABLE IF NOT EXISTS idx.chunks_fts USING fts5(
    text,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS idx.hype_vec USING vec0(
    hype_id        INTEGER PRIMARY KEY,
    +chunk_id      INTEGER,
    +question      TEXT,
    embedding_bin  BIT[%d],
    embedding_full FLOAT[%d]
);
]], binary_dim, embed_dim, binary_dim, embed_dim)
end

-- ── Meta access ─────────────────────────────────────────────────────────

--- Read a single value from `<schema>.meta`.
---
--- @param  conn    userdata  lsqlite3 connection.
--- @param  schema  string    `"main"` or `"idx"`.
--- @param  key     string
--- @return string?           The stored value, or nil when the key is absent.
function M.read_meta(conn, schema, key)
    local sql = string.format(
        "SELECT value FROM %s.meta WHERE key = ?", schema)
    local stmt = assert(conn:prepare(sql))
    stmt:bind_values(key)
    local rc = stmt:step()
    local v
    if rc == 100 then -- SQLITE_ROW
        v = stmt:get_value(0)
    end
    stmt:finalize()
    return v
end

--- Insert or overwrite a value in `<schema>.meta`.
---
--- @param  conn    userdata
--- @param  schema  string  `"main"` or `"idx"`.
--- @param  key     string
--- @param  value   string|number  Stored as text after `tostring`.
function M.write_meta(conn, schema, key, value)
    local sql = string.format([[
        INSERT INTO %s.meta(key, value) VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
    ]], schema)
    local stmt = assert(conn:prepare(sql))
    stmt:bind_values(key, tostring(value))
    assert(stmt:step() == 101) -- SQLITE_DONE
    stmt:finalize()
end

-- ── Bootstrap ───────────────────────────────────────────────────────────

--- Apply the chunks.db DDL idempotently and stamp the schema version.
--- A DB written by an older build has its missing tables created (every
--- CREATE is `IF NOT EXISTS`) and its stamp bumped. A DB written by a
--- newer build raises rather than risking corruption.
---
--- @param  conn  userdata  lsqlite3 connection.
--- @raise When the on-disk schema version is newer than this build.
function M.bootstrap_chunks(conn)
    assert(conn:exec(M.CHUNKS_DDL) == 0, "chunks.db DDL failed")

    local v = M.read_meta(conn, "main", "schema_version")
    if not v then
        M.write_meta(conn, "main", "schema_version", M.SCHEMA_VERSION)
    elseif tonumber(v) > M.SCHEMA_VERSION then
        error(string.format(
            "ion7-rag : chunks.db schema is newer than this build " ..
            "(file = %s, runtime = %d). Upgrade ion7-rag.",
            v, M.SCHEMA_VERSION))
    elseif tonumber(v) < M.SCHEMA_VERSION then
        M.write_meta(conn, "main", "schema_version", M.SCHEMA_VERSION)
    end
end

--- Apply the index.db DDL idempotently and validate runtime dimensions.
--- On first creation, stamps `schema_version`, `embed_dim`, and
--- `binary_dim` into `idx.meta`. On re-open, raises when the requested
--- dimensions don't match the stored ones — vec0 column widths cannot
--- be changed without rebuilding the index.
---
--- @param  conn  userdata  lsqlite3 connection.
--- @param  opts  table     `{ embed_dim, binary_dim }`.
--- @raise When the on-disk schema version is newer than this build, or
---        when the requested `embed_dim` / `binary_dim` differ from the
---        stored values.
function M.bootstrap_index(conn, opts)
    assert(conn:exec(M.index_ddl(opts)) == 0, "index.db DDL failed")

    local existing_version    = M.read_meta(conn, "idx", "schema_version")
    local existing_embed_dim  = M.read_meta(conn, "idx", "embed_dim")
    local existing_binary_dim = M.read_meta(conn, "idx", "binary_dim")

    if not existing_version then
        M.write_meta(conn, "idx", "schema_version", M.SCHEMA_VERSION)
        M.write_meta(conn, "idx", "embed_dim",      opts.embed_dim)
        M.write_meta(conn, "idx", "binary_dim",     opts.binary_dim)
        return
    end

    local existing_v_int = tonumber(existing_version)
    if existing_v_int > M.SCHEMA_VERSION then
        error(string.format(
            "ion7-rag : index.db schema is newer than this build " ..
            "(file = %s, runtime = %d). Upgrade ion7-rag.",
            existing_version, M.SCHEMA_VERSION))
    elseif existing_v_int < M.SCHEMA_VERSION then
        M.write_meta(conn, "idx", "schema_version", M.SCHEMA_VERSION)
    end
    if tonumber(existing_embed_dim) ~= opts.embed_dim then
        error(string.format(
            "ion7-rag : index.db embed_dim mismatch (file = %s, runtime = %d) — " ..
            "drop index.db and re-ingest, or open with embed_dim = %s",
            existing_embed_dim, opts.embed_dim, existing_embed_dim))
    end
    if tonumber(existing_binary_dim) ~= opts.binary_dim then
        error(string.format(
            "ion7-rag : index.db binary_dim mismatch (file = %s, runtime = %d) — " ..
            "drop index.db and re-ingest, or open with binary_dim = %s",
            existing_binary_dim, opts.binary_dim, existing_binary_dim))
    end
end

return M
