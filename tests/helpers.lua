--- @module tests.helpers
--- @author  ion7 / Ion7 Project Contributors
---
--- Shared scaffolding for the ion7-rag test suite. Mirrors the
--- ion7-core / ion7-llm helper modules — same env vars, same
--- skip-don't-fail contract.
---
--- Path bootstrap : tests are launched from the repo root, but the
--- in-tree sources live under `src/ion7/rag/`. We prepend the local
--- `src/?.lua` roots, AND we look for sibling checkouts of ion7-core,
--- ion7-llm and ion7-grammar so a developer working from
--- `_updates/ion7-rag/tests/run_all.sh` can pick up the matching
--- `_updates/ion7-{core,llm,grammar}/src/...` without a luarocks make
--- after every change.
---
--- Environment :
---   ION7_MODEL           Chat model GGUF for tests that drive an LLM
---                        (Contextual Retrieval, CRAG loop, eval judge).
---   ION7_EMBED_MODEL     Embedder GGUF (default Qwen3-Embedding-0.6B).
---   ION7_RERANK_MODEL    Cross-encoder reranker GGUF
---                        (default Qwen3-Reranker-0.6B).
---   ION7_CONTEXT_MODEL   Optional small contextualizer GGUF for the
---                        Contextual Retrieval Pool ; falls back to
---                        ION7_MODEL when not set.
---   ION7_RAG_CORPUS      Path to a directory of test documents
---                        (the Idantic synthetic dataset, etc.).
---   ION7_GPU_LAYERS      Override n_gpu_layers (default 0).
---   ION7_CORE_SRC        Override ion7-core source root.
---   ION7_LLM_SRC         Override ion7-llm source root.
---   ION7_GRAMMAR_SRC     Override ion7-grammar source root.

-- ── package.path bootstrap ────────────────────────────────────────────────

local function _add_path(prefix)
    local extras = prefix .. "/?.lua;" .. prefix .. "/?/init.lua"
    if not package.path:find(extras, 1, true) then
        package.path = extras .. ";" .. package.path
    end
end

local function _probe_sibling(env_var, marker_path, candidates)
    local override = os.getenv(env_var)
    if override and override ~= "" then
        _add_path(override)
        return true
    end
    for _, candidate in ipairs(candidates) do
        local f = io.open(candidate .. "/" .. marker_path, "r")
        if f then
            f:close()
            _add_path(candidate)
            return true
        end
    end
    return false
end

local function _bootstrap_paths()
    -- Local ion7-rag sources first.
    _add_path("./src")

    -- Sibling ion7-core sources.
    _probe_sibling("ION7_CORE_SRC", "ion7/core/init.lua", {
        "../ion7-core/src",
        "../../ion7-core/src",
    })

    -- Sibling ion7-llm sources.
    _probe_sibling("ION7_LLM_SRC", "ion7/llm/init.lua", {
        "../ion7-llm/src",
        "../../ion7-llm/src",
    })

    -- Sibling ion7-grammar sources.
    _probe_sibling("ION7_GRAMMAR_SRC", "ion7/grammar/init.lua", {
        "../ion7-grammar/src",
        "../../ion7-grammar/src",
    })
end

_bootstrap_paths()

local M = {}

-- ── Environment helpers ───────────────────────────────────────────────────

local function _env(name)
    local v = os.getenv(name)
    if v == nil or v == "" then return nil end
    return v
end

M._env = _env

function M.model_path()           return _env("ION7_MODEL") end
function M.embed_model_path()     return _env("ION7_EMBED_MODEL") end
function M.rerank_model_path()    return _env("ION7_RERANK_MODEL") end
function M.context_model_path()
    return _env("ION7_CONTEXT_MODEL") or _env("ION7_MODEL")
end
function M.corpus_path()          return _env("ION7_RAG_CORPUS") end

function M.gpu_layers()
    return tonumber(_env("ION7_GPU_LAYERS") or "0") or 0
end

local function _require_env(T, var, hint)
    local v = _env(var)
    if not v then
        T.skip("(this whole file)",
            var .. " not set — " .. (hint or ("export " .. var .. "=...")))
        T.summary()
        os.exit(0)
    end
    return v
end

function M.require_model(T)
    return _require_env(T, "ION7_MODEL",
        "export ION7_MODEL=/path/to/chat.gguf")
end

function M.require_embed_model(T)
    return _require_env(T, "ION7_EMBED_MODEL",
        "export ION7_EMBED_MODEL=/path/to/embed.gguf")
end

function M.require_rerank_model(T)
    return _require_env(T, "ION7_RERANK_MODEL",
        "export ION7_RERANK_MODEL=/path/to/reranker.gguf")
end

function M.require_corpus(T)
    return _require_env(T, "ION7_RAG_CORPUS",
        "export ION7_RAG_CORPUS=/path/to/dataset/")
end

-- ── Filesystem helpers ────────────────────────────────────────────────────

function M.tmpfile(basename)
    local dir = _env("TMPDIR") or _env("TEMP") or "/tmp"
    return dir .. "/" .. basename
end

function M.try_remove(path) pcall(os.remove, path) end

--- Build a unique pair of (chunks.db, index.db) paths under TMPDIR for a
--- single test file. Both files are best-effort removed — call this once
--- per test file at boot, not once per `T.test`.
function M.tmp_db_pair(tag)
    local stem = string.format("ion7-rag-test-%s-%d", tag or "x",
        math.floor(os.time() * 1000) % 1e6)
    local chunks = M.tmpfile(stem .. ".chunks.db")
    local index  = M.tmpfile(stem .. ".index.db")
    M.try_remove(chunks)
    M.try_remove(index)
    return chunks, index
end

-- ── sqlite-vec discovery ──────────────────────────────────────────────────

local _SQLITE_VEC_PROBE = {
    "/usr/local/lib/vec0.so",
    "/usr/lib64/vec0.so",
    "/usr/lib/vec0.so",
    "/tmp/sqlite-vec/dist/vec0.so",
    "/opt/sqlite-vec/vec0.so",
}

--- Locate the sqlite-vec shared object. Honours
--- `ION7_RAG_SQLITE_VEC_PATH` first, then probes a small list of common
--- install locations. Returns the path or nil.
function M.find_sqlite_vec()
    local override = _env("ION7_RAG_SQLITE_VEC_PATH")
    if override and override ~= "" then
        local f = io.open(override, "r")
        if f then f:close() ; return override end
    end
    for _, candidate in ipairs(_SQLITE_VEC_PROBE) do
        local f = io.open(candidate, "r")
        if f then f:close() ; return candidate end
    end
    return nil
end

--- Skip-don't-fail variant : returns the resolved path or skips the
--- whole file when sqlite-vec cannot be located.
function M.require_sqlite_vec(T)
    local path = M.find_sqlite_vec()
    if not path then
        T.skip("(this whole file)",
            "sqlite-vec not found — set ION7_RAG_SQLITE_VEC_PATH to vec0.so " ..
            "(see ion7-rag/INSTALL.md for build instructions)")
        T.summary()
        os.exit(0)
    end
    return path
end

-- ── lsqlite3 availability ─────────────────────────────────────────────────

--- Skip-don't-fail variant : require lsqlite3, propagating the package
--- error verbatim if absent.
function M.require_lsqlite3(T)
    local ok, sqlite3 = pcall(require, "lsqlite3")
    if not ok then
        T.skip("(this whole file)",
            "lsqlite3 not installed — luarocks install --local lsqlite3")
        T.summary()
        os.exit(0)
    end
    return sqlite3
end

-- ── Backend bring-up ──────────────────────────────────────────────────────

--- Require ion7-core (libllama, optional bridge), skipping the whole file
--- gracefully when the libraries cannot be loaded.
function M.require_backend(T)
    local ok, ion7 = pcall(require, "ion7.core")
    if not ok then
        T.skip("(this whole file)",
            "ion7.core failed to load — build vendor/llama.cpp + bridge first " ..
            "(or set ION7_CORE_SRC). Underlying error: " ..
            tostring(ion7):sub(1, 200))
        T.summary()
        os.exit(0)
    end

    local init_ok, err = pcall(ion7.init, { log_level = 0 })
    if not init_ok then
        T.skip("(this whole file)",
            "ion7.init() failed: " .. tostring(err):sub(1, 200))
        T.summary()
        os.exit(0)
    end

    return ion7
end

return M
