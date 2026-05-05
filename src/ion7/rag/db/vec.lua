--- @module ion7.rag.db.vec
--- @author  ion7 / Ion7 Project Contributors
---
--- sqlite-vec dense vector tier (`idx.chunks_vec`).
---
--- Two vector columns per row, both keyed by `chunk_id` matching
--- `chunks.id` :
---
---   embedding_bin   BIT[binary_dim]   shortlist tier, Hamming distance
---   embedding_full  FLOAT[embed_dim]  rerank tier, L2 distance
---
--- The binary column holds the first `binary_dim` dimensions of the
--- full embedding (Matryoshka truncation), sign-quantised by
--- sqlite-vec's `vec_quantize_binary`. The MRL property of modern
--- embedders (Qwen3-Embedding, Jina v3+, Nomic v1.5+) keeps that
--- truncation discriminative enough for shortlist retrieval.
---
--- Vectors are passed as JSON text (`[0.1, -0.2, ...]`) and
--- materialised through `vec_f32()` on the SQL side. The JSON path is
--- chosen for clarity ; a BLOB-packed FFI path can replace it where
--- profiling shows insertion throughput as a bottleneck.

local M = {}

-- ── Internal helpers ────────────────────────────────────────────────────

--- Render a Lua array of floats as a compact JSON array. `%.7g` keeps
--- ~7 significant decimal digits, enough to round-trip an fp32 value
--- without ballooning the wire format.
local function _to_json_array(vec, n)
    n = n or #vec
    local parts = {}
    for i = 1, n do
        parts[i] = string.format("%.7g", vec[i])
    end
    return "[" .. table.concat(parts, ",") .. "]"
end

local function _check_dim(vec, expected, label)
    if #vec ~= expected then
        error(string.format(
            "ion7.rag.db.vec : %s expected %d dimensions, got %d",
            label, expected, #vec), 3)
    end
end

-- ── Insert ──────────────────────────────────────────────────────────────

-- vec0's `chunk_id` column is strictly typed as INTEGER. lsqlite3 binds
-- every Lua number through `sqlite3_bind_double`, so we cast on the SQL
-- side to land back in SQLITE_INTEGER for vec0's type check.
local _DELETE_SQL = "DELETE FROM idx.chunks_vec WHERE chunk_id = CAST(? AS INTEGER)"
local _INSERT_SQL = [[
    INSERT INTO idx.chunks_vec(chunk_id, embedding_bin, embedding_full)
    VALUES (CAST(? AS INTEGER),
            vec_quantize_binary(vec_f32(?)),
            vec_f32(?))
]]

--- Insert or replace one vector row.
---
--- vec0 virtual tables don't support UPDATE, so an upsert is a
--- DELETE-then-INSERT inside a single statement pair.
---
--- @param  h          ion7.rag.db.Handle
--- @param  chunk_id   integer   Must match an existing `chunks.id`.
--- @param  embedding  number[]  Length must equal `h:embed_dim()`. The
---                     first `h:binary_dim()` dimensions feed the
---                     binary tier ; the full vector feeds the fp32 tier.
--- @raise When the vector dimension doesn't match `h:embed_dim()` or
---        when the INSERT fails.
function M.upsert(h, chunk_id, embedding)
    _check_dim(embedding, h:embed_dim(), "upsert embedding")

    local full_json = _to_json_array(embedding, h:embed_dim())
    local bin_json  = _to_json_array(embedding, h:binary_dim())

    local del = h:prepare(_DELETE_SQL)
    del:bind_values(chunk_id)
    del:step()
    del:finalize()

    local ins = h:prepare(_INSERT_SQL)
    ins:bind_values(chunk_id, bin_json, full_json)
    local rc = ins:step()
    ins:finalize()
    if rc ~= 101 then -- SQLITE_DONE
        error("vec.upsert : step failed (rc=" .. tostring(rc) ..
              ", " .. tostring(h:conn():errmsg()) .. ")", 2)
    end
end

--- Bulk-upsert vector rows. Single SQLite transaction, two reused
--- prepared statements.
---
--- @param  h     ion7.rag.db.Handle
--- @param  rows  table[]  `{ { chunk_id = integer, embedding = number[] }, ... }`
--- @raise When any row misses a required field, has a wrong-dimension
---        embedding, or when an INSERT fails.
function M.upsert_many(h, rows)
    h:transaction(function()
        local del = h:prepare(_DELETE_SQL)
        local ins = h:prepare(_INSERT_SQL)
        for i, row in ipairs(rows) do
            assert(row.chunk_id and row.embedding,
                   "vec.upsert_many[" .. i .. "] : missing chunk_id / embedding")
            _check_dim(row.embedding, h:embed_dim(),
                "upsert_many[" .. i .. "] embedding")

            del:bind_values(row.chunk_id) ; del:step() ; del:reset()

            ins:bind_values(
                row.chunk_id,
                _to_json_array(row.embedding, h:binary_dim()),
                _to_json_array(row.embedding, h:embed_dim())
            )
            local rc = ins:step()
            if rc ~= 101 then
                ins:finalize() ; del:finalize()
                error("vec.upsert_many[" .. i .. "] : rc=" .. tostring(rc), 2)
            end
            ins:reset()
        end
        ins:finalize()
        del:finalize()
    end)
end

--- Delete one vector row.
---
--- @param  h         ion7.rag.db.Handle
--- @param  chunk_id  integer
--- @return integer  Number of rows the DELETE affected (0 or 1).
function M.delete(h, chunk_id)
    local stmt = h:prepare(_DELETE_SQL)
    stmt:bind_values(chunk_id)
    stmt:step()
    stmt:finalize()
    return h:changes()
end

--- Total number of rows currently in `idx.chunks_vec`.
--- @param  h  ion7.rag.db.Handle
--- @return integer
function M.count(h)
    local stmt = h:prepare("SELECT COUNT(*) FROM idx.chunks_vec")
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n
end

-- ── Search ──────────────────────────────────────────────────────────────

--- KNN search over the binary tier. Uses Hamming distance ; the fast
--- path for shortlist scenarios.
---
--- @param  h      ion7.rag.db.Handle
--- @param  query  number[]  Length must equal `h:embed_dim()` ; it is
---                  truncated to `h:binary_dim()` and sign-quantised
---                  before the MATCH.
--- @param  k      integer   Number of hits to return (`k > 0`).
--- @return Hit[]            `{ { chunk_id, distance }, ... }` ordered
---                  by ascending distance.
--- @raise When the query has the wrong dimension or `k <= 0`.
function M.knn_binary(h, query, k)
    _check_dim(query, h:embed_dim(), "knn_binary query")
    assert(k and k > 0, "knn_binary : k must be > 0")

    local q_json = _to_json_array(query, h:binary_dim())

    local stmt = h:prepare([[
        SELECT chunk_id, distance
        FROM idx.chunks_vec
        WHERE embedding_bin MATCH vec_quantize_binary(vec_f32(?))
          AND k = ?
        ORDER BY distance
    ]])
    stmt:bind_values(q_json, k)

    local hits = {}
    while stmt:step() == 100 do -- SQLITE_ROW
        hits[#hits + 1] = {
            chunk_id = stmt:get_value(0),
            distance = stmt:get_value(1),
        }
    end
    stmt:finalize()
    return hits
end

--- KNN search over the fp32 tier. Uses L2 distance ; the high-fidelity
--- path, suitable as a standalone search or as a rerank pass over the
--- binary tier's shortlist.
---
--- @param  h      ion7.rag.db.Handle
--- @param  query  number[]  Length must equal `h:embed_dim()`.
--- @param  k      integer   Number of hits to return (`k > 0`).
--- @return Hit[]            `{ { chunk_id, distance }, ... }` ordered
---                  by ascending distance.
--- @raise When the query has the wrong dimension or `k <= 0`.
function M.knn_full(h, query, k)
    _check_dim(query, h:embed_dim(), "knn_full query")
    assert(k and k > 0, "knn_full : k must be > 0")

    local q_json = _to_json_array(query, h:embed_dim())

    local stmt = h:prepare([[
        SELECT chunk_id, distance
        FROM idx.chunks_vec
        WHERE embedding_full MATCH vec_f32(?)
          AND k = ?
        ORDER BY distance
    ]])
    stmt:bind_values(q_json, k)

    local hits = {}
    while stmt:step() == 100 do
        hits[#hits + 1] = {
            chunk_id = stmt:get_value(0),
            distance = stmt:get_value(1),
        }
    end
    stmt:finalize()
    return hits
end

return M
