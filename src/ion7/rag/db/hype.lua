--- @module ion7.rag.db.hype
--- @author  ion7 / Ion7 Project Contributors
---
--- HyPE (Hypothetical Prompt Embeddings) vector store. Each chunk
--- spawns N hypothetical-question vectors at ingest time ; this module
--- handles their CRUD on `idx.hype_vec`.
---
--- Schema (vec0 with auxiliary columns) :
---
---   hype_id        INTEGER PRIMARY KEY
---   +chunk_id      INTEGER             back-reference to `chunks.id`
---   +question      TEXT                raw question text
---   embedding_bin  BIT[binary_dim]
---   embedding_full FLOAT[embed_dim]
---
--- A KNN MATCH returns `chunk_id`, `question`, and `distance` in a
--- single query, so `ion7.rag.retrieve.search` folds HyPE hits into
--- the dense candidate set without a JOIN.

local M = {}

-- ── Internal helpers ────────────────────────────────────────────────────

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
            "ion7.rag.db.hype : %s expected %d dimensions, got %d",
            label, expected, #vec), 3)
    end
end

-- vec0 strictly types INTEGER columns, so we CAST on the SQL side ;
-- same rationale as `ion7.rag.db.vec`.
local _DELETE_BY_HYPE_ID  = "DELETE FROM idx.hype_vec WHERE hype_id = CAST(? AS INTEGER)"
local _DELETE_BY_CHUNK_ID = "DELETE FROM idx.hype_vec WHERE chunk_id = CAST(? AS INTEGER)"
local _INSERT_SQL = [[
    INSERT INTO idx.hype_vec(hype_id, chunk_id, question, embedding_bin, embedding_full)
    VALUES (CAST(? AS INTEGER),
            CAST(? AS INTEGER),
            ?,
            vec_quantize_binary(vec_f32(?)),
            vec_f32(?))
]]

-- ── Insert ──────────────────────────────────────────────────────────────

--- Insert or replace one HyPE row.
---
--- @param  h          ion7.rag.db.Handle
--- @param  hype_id    integer   Caller-assigned primary key. Re-using
---                     an existing id replaces that row in place.
--- @param  chunk_id   integer   Back-reference to `chunks.id`.
--- @param  question   string    The hypothetical question text.
--- @param  embedding  number[]  Length must equal `h:embed_dim()`.
--- @raise When the embedding has the wrong dimension or the INSERT fails.
function M.upsert(h, hype_id, chunk_id, question, embedding)
    _check_dim(embedding, h:embed_dim(), "upsert embedding")

    local del = h:prepare(_DELETE_BY_HYPE_ID)
    del:bind_values(hype_id) ; del:step() ; del:finalize()

    local full_json = _to_json_array(embedding, h:embed_dim())
    local bin_json  = _to_json_array(embedding, h:binary_dim())

    local ins = h:prepare(_INSERT_SQL)
    ins:bind_values(hype_id, chunk_id, question, bin_json, full_json)
    local rc = ins:step()
    ins:finalize()
    if rc ~= 101 then
        error("hype.upsert : step failed (rc=" .. tostring(rc) ..
              ", " .. tostring(h:conn():errmsg()) .. ")", 2)
    end
end

--- Bulk-upsert HyPE rows. Single SQLite transaction, two reused
--- prepared statements.
---
--- @param  h     ion7.rag.db.Handle
--- @param  rows  table[]  `{ { hype_id, chunk_id, question, embedding }, ... }`
--- @raise When any row misses a required field, has a wrong-dimension
---        embedding, or when an INSERT fails.
function M.upsert_many(h, rows)
    h:transaction(function()
        local del = h:prepare(_DELETE_BY_HYPE_ID)
        local ins = h:prepare(_INSERT_SQL)
        for i, row in ipairs(rows) do
            assert(row.hype_id and row.chunk_id and row.question and row.embedding,
                "hype.upsert_many[" .. i .. "] : missing required field")
            _check_dim(row.embedding, h:embed_dim(),
                "upsert_many[" .. i .. "] embedding")

            del:bind_values(row.hype_id) ; del:step() ; del:reset()
            ins:bind_values(
                row.hype_id, row.chunk_id, row.question,
                _to_json_array(row.embedding, h:binary_dim()),
                _to_json_array(row.embedding, h:embed_dim())
            )
            local rc = ins:step()
            if rc ~= 101 then
                ins:finalize() ; del:finalize()
                error("hype.upsert_many[" .. i .. "] : rc=" .. tostring(rc), 2)
            end
            ins:reset()
        end
        ins:finalize()
        del:finalize()
    end)
end

--- Delete every HyPE row that belongs to `chunk_id`. Called when a
--- chunk is being re-ingested or when a doc is wiped from the index.
---
--- @param  h         ion7.rag.db.Handle
--- @param  chunk_id  integer
--- @return integer  Number of rows the DELETE affected.
function M.delete_for_chunk(h, chunk_id)
    local stmt = h:prepare(_DELETE_BY_CHUNK_ID)
    stmt:bind_values(chunk_id)
    stmt:step()
    stmt:finalize()
    return h:changes()
end

--- Total number of rows currently in `idx.hype_vec`.
--- @param  h  ion7.rag.db.Handle
--- @return integer
function M.count(h)
    local stmt = h:prepare("SELECT COUNT(*) FROM idx.hype_vec")
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n
end

-- ── Search ──────────────────────────────────────────────────────────────

--- KNN search over the binary tier. The auxiliary columns
--- (`chunk_id`, `question`) ride along with the MATCH so each hit
--- already carries the parent chunk back-reference.
---
--- @param  h      ion7.rag.db.Handle
--- @param  query  number[]  Length must equal `h:embed_dim()` ; truncated
---                  and quantised to the binary tier internally.
--- @param  k      integer   `k > 0`.
--- @return table[]          `{ { hype_id, chunk_id, question, distance }, ... }`
---                  ordered by ascending Hamming distance.
--- @raise When the query has the wrong dimension or `k <= 0`.
function M.knn_binary(h, query, k)
    _check_dim(query, h:embed_dim(), "knn_binary query")
    assert(k and k > 0, "knn_binary : k must be > 0")

    local q_json = _to_json_array(query, h:binary_dim())

    local stmt = h:prepare([[
        SELECT hype_id, chunk_id, question, distance
        FROM idx.hype_vec
        WHERE embedding_bin MATCH vec_quantize_binary(vec_f32(?))
          AND k = ?
        ORDER BY distance
    ]])
    stmt:bind_values(q_json, k)

    local hits = {}
    while stmt:step() == 100 do
        hits[#hits + 1] = {
            hype_id  = stmt:get_value(0),
            chunk_id = stmt:get_value(1),
            question = stmt:get_value(2),
            distance = stmt:get_value(3),
        }
    end
    stmt:finalize()
    return hits
end

--- KNN search over the fp32 tier. Same shape as `knn_binary` but uses
--- L2 distance on the full-precision vectors.
---
--- @param  h      ion7.rag.db.Handle
--- @param  query  number[]
--- @param  k      integer
--- @return table[]  `{ { hype_id, chunk_id, question, distance }, ... }`
--- @raise When the query has the wrong dimension or `k <= 0`.
function M.knn_full(h, query, k)
    _check_dim(query, h:embed_dim(), "knn_full query")
    assert(k and k > 0, "knn_full : k must be > 0")

    local q_json = _to_json_array(query, h:embed_dim())

    local stmt = h:prepare([[
        SELECT hype_id, chunk_id, question, distance
        FROM idx.hype_vec
        WHERE embedding_full MATCH vec_f32(?)
          AND k = ?
        ORDER BY distance
    ]])
    stmt:bind_values(q_json, k)

    local hits = {}
    while stmt:step() == 100 do
        hits[#hits + 1] = {
            hype_id  = stmt:get_value(0),
            chunk_id = stmt:get_value(1),
            question = stmt:get_value(2),
            distance = stmt:get_value(3),
        }
    end
    stmt:finalize()
    return hits
end

--- Collapse a list of HyPE hits to one entry per `chunk_id`, keeping
--- the smallest distance for each. The retrieval glue calls this
--- before feeding HyPE hits into fusion so a single chunk never
--- contributes multiple rows under the same source.
---
--- @param  hype_hits  table[]  Output of `knn_binary` or `knn_full`.
--- @return table[]              `{ { chunk_id, distance }, ... }` sorted
---                               by ascending distance.
function M.collapse_to_chunks(hype_hits)
    local best = {}
    for _, hit in ipairs(hype_hits) do
        local prev = best[hit.chunk_id]
        if not prev or hit.distance < prev then
            best[hit.chunk_id] = hit.distance
        end
    end
    local out = {}
    for cid, dist in pairs(best) do
        out[#out + 1] = { chunk_id = cid, distance = dist }
    end
    table.sort(out, function(a, b) return a.distance < b.distance end)
    return out
end

return M
