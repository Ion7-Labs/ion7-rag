--- @module ion7.rag.db
--- @author  ion7 / Ion7 Project Contributors
---
--- Two-file SQLite store for ion7-rag.
---
---   chunks.db   canonical text + metadata + provenance, source of truth.
---   index.db    sqlite-vec dense vectors + FTS5 BM25 + HyPE-question
---               vectors, ATTACHed as `idx`, disposable and rebuildable
---               from chunks.db.
---
--- The `Handle` returned by `db.open` opens both files, loads the
--- sqlite-vec extension once, ATTACHes index.db, applies the schema
--- bootstrap, and exposes connection-level primitives (`:exec`,
--- `:prepare`, `:transaction`, `:close`). All higher-level CRUD lives
--- in sibling sub-modules :
---
---   ion7.rag.db.chunks   docs + chunks tables on chunks.db (main).
---   ion7.rag.db.vec      vec0 chunk vectors on idx.chunks_vec.
---   ion7.rag.db.lex      FTS5 BM25 on idx.chunks_fts.
---   ion7.rag.db.hype     HyPE-question vectors on idx.hype_vec.
---
--- @usage
---   local db = require "ion7.rag.db"
---   local h = db.open({
---       chunks_path = "./data/chunks.db",
---       index_path  = "./data/index.db",
---       embed_dim   = 1024,
---       binary_dim  = 192,
---   })
---   -- ... use h with the sub-modules ...
---   h:close()

local sqlite3 = require "lsqlite3"

local schema = require "ion7.rag.db.schema"

local M = {}

-- ── Handle class ────────────────────────────────────────────────────────

local Handle = {}
Handle.__index = Handle

M.Handle = Handle

-- ── Required-opts validation ────────────────────────────────────────────

local function _require(opts, key)
    if opts[key] == nil then
        error("ion7.rag.db.open : opts." .. key ..
              " is required (no hardcoded defaults)", 3)
    end
    return opts[key]
end

-- ── sqlite-vec extension loading ────────────────────────────────────────

--- Load the sqlite-vec C extension into `conn`. Resolution order :
---
---   1. `opts.sqlite_vec_path`                 explicit absolute path
---   2. `ION7_RAG_SQLITE_VEC_PATH` env var
---   3. SQLite's own extension search ("sqlite-vec" by name)
---
--- Failure to load surfaces a single clear error rather than degrading
--- to a no-op. Extension loading is disabled again immediately after
--- sqlite-vec is in place, so a later adversarial
--- `SELECT load_extension(...)` cannot pull arbitrary shared objects
--- into the connection.
local function _load_vec_extension(conn, opts)
    -- lsqlite3 builds vary on whether `enable_load_extension` is
    -- exposed as a method. When absent, `load_extension` triggers the
    -- underlying SQLite C call transparently.
    if type(conn.enable_load_extension) == "function" then
        conn:enable_load_extension(true)
    end

    local path = opts.sqlite_vec_path
              or os.getenv("ION7_RAG_SQLITE_VEC_PATH")

    local rc, err
    if path and path ~= "" then
        rc, err = conn:load_extension(path)
    else
        rc, err = conn:load_extension("sqlite-vec")
    end

    if type(conn.enable_load_extension) == "function" then
        conn:enable_load_extension(false)
    end

    if not rc then
        error("ion7.rag.db : failed to load sqlite-vec extension : " ..
              tostring(err) ..
              " — set opts.sqlite_vec_path or ION7_RAG_SQLITE_VEC_PATH to " ..
              "the absolute path of the sqlite-vec shared object.", 0)
    end
end

-- ── Open ────────────────────────────────────────────────────────────────

--- Open both databases, ATTACH index.db as `idx`, load the sqlite-vec
--- extension, apply schema migrations, and return a `Handle` to the
--- combined connection. Both files are created on the spot if they
--- don't yet exist.
---
--- WAL journaling and `synchronous = NORMAL` are enabled by default,
--- foreign keys are turned on. None of these PRAGMAs survive process
--- restart on their own — `db.open` re-applies them every time.
---
--- @param  opts table {
---     chunks_path      string   Path to chunks.db (main).
---     index_path       string   Path to index.db (attached as `idx`).
---     embed_dim        integer  Full-precision vector width (e.g. 1024).
---     binary_dim       integer  Binary-quantised shortlist width
---                       (e.g. 192). Validated against the on-disk
---                       value on re-open ; cannot be changed without
---                       rebuilding the index.
---     sqlite_vec_path  string?  Absolute path to the sqlite-vec
---                       extension. Falls back to
---                       `ION7_RAG_SQLITE_VEC_PATH` and then to the
---                       SQLite default extension search.
---     wal              bool?    Set to `false` to skip WAL setup.
---                       Default `true`.
--- }
--- @return ion7.rag.db.Handle
--- @raise When any required opt is missing, when sqlite3.open or ATTACH
---        fails, or when sqlite-vec cannot be loaded.
function M.open(opts)
    opts = opts or {}
    local chunks_path = _require(opts, "chunks_path")
    local index_path  = _require(opts, "index_path")
    local embed_dim   = _require(opts, "embed_dim")
    local binary_dim  = _require(opts, "binary_dim")

    local conn, code, msg = sqlite3.open(chunks_path)
    if not conn then
        error(string.format(
            "ion7.rag.db : sqlite3.open('%s') failed : [%s] %s",
            chunks_path, tostring(code), tostring(msg)), 0)
    end

    local self = setmetatable({
        _conn        = conn,
        _chunks_path = chunks_path,
        _index_path  = index_path,
        _embed_dim   = embed_dim,
        _binary_dim  = binary_dim,
        _closed      = false,
    }, Handle)

    if opts.wal ~= false then
        self:exec("PRAGMA journal_mode = WAL")
    end
    self:exec("PRAGMA synchronous = NORMAL")
    self:exec("PRAGMA foreign_keys = ON")

    _load_vec_extension(conn, opts)

    -- The path is bound rather than concatenated so a path containing
    -- a single quote cannot break the ATTACH statement.
    local stmt = conn:prepare("ATTACH DATABASE ? AS idx")
    stmt:bind_values(index_path)
    local rc = stmt:step()
    stmt:finalize()
    if rc ~= 101 then -- SQLITE_DONE
        self:close()
        error(string.format(
            "ion7.rag.db : ATTACH '%s' AS idx failed : [%s] %s",
            index_path, tostring(rc), tostring(conn:errmsg())), 0)
    end

    -- The attached schema needs its own PRAGMA pass.
    self:exec("PRAGMA idx.synchronous = NORMAL")

    schema.bootstrap_chunks(conn)
    schema.bootstrap_index(conn, {
        embed_dim  = embed_dim,
        binary_dim = binary_dim,
    })

    return self
end

-- ── Connection primitives ───────────────────────────────────────────────

--- Execute a non-query SQL string.
--- @param  sql  string
--- @raise When the underlying `sqlite3_exec` returns a non-zero code ;
---        the error message carries the offending SQL.
function Handle:exec(sql)
    local rc = self._conn:exec(sql)
    if rc ~= 0 then
        error(string.format(
            "ion7.rag.db : exec failed : [%s] %s\n  SQL : %s",
            tostring(rc), tostring(self._conn:errmsg()), sql), 2)
    end
end

--- Compile a SQL string into a prepared statement. The caller is
--- responsible for `:finalize()` once they're done with the statement.
--- @param  sql  string
--- @return userdata  lsqlite3 prepared statement.
--- @raise When the SQL fails to compile.
function Handle:prepare(sql)
    local stmt, err = self._conn:prepare(sql)
    if not stmt then
        error(string.format(
            "ion7.rag.db : prepare failed : %s\n  SQL : %s",
            tostring(err or self._conn:errmsg()), sql), 2)
    end
    return stmt
end

--- Run `fn(self)` inside a SQLite transaction. The transaction commits
--- when `fn` returns and rolls back when it raises ; the original
--- error is then re-raised to the caller.
--- @param  fn  function(Handle)
--- @return true  On commit.
function Handle:transaction(fn)
    self:exec("BEGIN")
    local ok, err = pcall(fn, self)
    if ok then
        self:exec("COMMIT")
        return true
    end
    self:exec("ROLLBACK")
    error(err, 2)
end

--- Prepare `sql`, run `fn(stmt)`, finalise the statement. The
--- statement is finalised whether `fn` returns or raises ; the
--- original error is re-raised.
function Handle:with_stmt(sql, fn)
    local stmt = self:prepare(sql)
    local ok, err = pcall(fn, stmt)
    stmt:finalize()
    if not ok then error(err, 2) end
end

--- Rowid of the most recent successful `INSERT`.
--- @return integer
function Handle:last_insert_rowid()
    return self._conn:last_insert_rowid()
end

--- Number of rows affected by the most recent statement.
--- @return integer
function Handle:changes()
    return self._conn:changes()
end

--- Underlying lsqlite3 connection. Exposed for advanced uses ; the
--- sub-modules in this namespace use it for direct `:nrows` iteration.
--- @return userdata
function Handle:conn() return self._conn end

--- The full-precision embedding dimension this handle was opened with.
--- Used by `db.vec` and `db.hype` to validate vectors at insert time.
--- @return integer
function Handle:embed_dim()  return self._embed_dim end

--- The binary-tier embedding dimension this handle was opened with.
--- @return integer
function Handle:binary_dim() return self._binary_dim end

--- Close the underlying connection. Idempotent. SQLite detaches
--- attached databases automatically when the main connection closes,
--- so no explicit DETACH is needed (and DETACH while statements are
--- pending would fail anyway — lsqlite3 finalises them during close).
function Handle:close()
    if self._closed then return end
    self._closed = true
    self._conn:close()
    self._conn = nil
end

return M
