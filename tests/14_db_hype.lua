#!/usr/bin/env luajit
--- @module tests.14_db_hype
---
--- HyPE vector store CRUD on the v2 schema. Synthetic vectors so no
--- embedder is needed.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.db.hype — vec0 + aux-column HyPE store")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db   = require "ion7.rag.db"
local hype = require "ion7.rag.db.hype"

local EMBED_DIM, BINARY_DIM = 32, 16

local cp, ip = H.tmp_db_pair("14")
local h = db.open({
    chunks_path     = cp,
    index_path      = ip,
    embed_dim       = EMBED_DIM,
    binary_dim      = BINARY_DIM,
    sqlite_vec_path = sqlite_vec_path,
})

local function cleanup()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
end

-- ── Synthetic vectors ──────────────────────────────────────────────────

math.randomseed(14)

local function rand_unit(dim)
    local v, s = {}, 0
    for i = 1, dim do v[i] = math.random()*2 - 1 ; s = s + v[i]*v[i] end
    local n = math.sqrt(s)
    for i = 1, dim do v[i] = v[i] / n end
    return v
end

local function perturb(seed, noise)
    local v, s = {}, 0
    for i = 1, #seed do
        v[i] = seed[i] + (math.random()*2 - 1)*noise
        s = s + v[i]*v[i]
    end
    local n = math.sqrt(s)
    for i = 1, #seed do v[i] = v[i] / n end
    return v
end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("upsert + count round-trip", function()
    hype.upsert(h, 1,   100, "What is the delivery window?",   rand_unit(EMBED_DIM))
    hype.upsert(h, 2,   100, "How fast is shipping?",           rand_unit(EMBED_DIM))
    hype.upsert(h, 3,   200, "Who pays for return shipping?",   rand_unit(EMBED_DIM))
    T.eq(hype.count(h), 3)
end)

T.test("upsert is idempotent on hype_id", function()
    hype.upsert(h, 1, 100, "rephrased", rand_unit(EMBED_DIM))
    T.eq(hype.count(h), 3)
end)

T.test("knn_full returns chunk_id + question + distance", function()
    local seed = rand_unit(EMBED_DIM)
    hype.upsert(h, 50, 500, "seeded question", seed)
    hype.upsert(h, 51, 501, "near question 1", perturb(seed, 0.4))
    hype.upsert(h, 52, 502, "near question 2", perturb(seed, 0.6))

    local hits = hype.knn_full(h, seed, 3)
    T.eq(#hits, 3)
    T.eq(hits[1].chunk_id, 500, "seed self should top")
    T.is_type(hits[1].question, "string")
    T.gte(hits[2].distance, hits[1].distance)
end)

T.test("knn_binary returns the same shape", function()
    local q = rand_unit(EMBED_DIM)
    local hits = hype.knn_binary(h, q, 3)
    T.eq(#hits, 3)
    T.is_type(hits[1].chunk_id, "number")
    T.is_type(hits[1].question, "string")
end)

T.test("collapse_to_chunks dedupes by chunk_id keeping min distance", function()
    local raw = {
        { hype_id = 1, chunk_id = 100, distance = 0.5 },
        { hype_id = 2, chunk_id = 100, distance = 0.2 },  -- better
        { hype_id = 3, chunk_id = 200, distance = 0.3 },
    }
    local collapsed = hype.collapse_to_chunks(raw)
    T.eq(#collapsed, 2)
    -- Sorted ASC by distance ; chunk 100 wins on its better hype_id 2.
    T.eq(collapsed[1].chunk_id, 100)
    T.eq(collapsed[1].distance, 0.2)
    T.eq(collapsed[2].chunk_id, 200)
end)

T.test("delete_for_chunk removes every hype row of a chunk", function()
    -- Plant several hype rows for chunk 999.
    hype.upsert(h, 900, 999, "q1", rand_unit(EMBED_DIM))
    hype.upsert(h, 901, 999, "q2", rand_unit(EMBED_DIM))
    hype.upsert(h, 902, 999, "q3", rand_unit(EMBED_DIM))
    local before = hype.count(h)
    local removed = hype.delete_for_chunk(h, 999)
    T.eq(removed, 3)
    T.eq(hype.count(h), before - 3)
end)

T.test("upsert_many wraps a transaction and accepts a batch", function()
    local rows = {}
    for i = 1, 20 do
        rows[i] = {
            hype_id   = 5000 + i,
            chunk_id  = 700 + (i % 5),
            question  = "bulk q " .. i,
            embedding = rand_unit(EMBED_DIM),
        }
    end
    hype.upsert_many(h, rows)
    -- Each chunk_id 700..704 has 4 hype rows now.
    local q = rand_unit(EMBED_DIM)
    local raw = hype.knn_full(h, q, 50)
    local collapsed = hype.collapse_to_chunks(raw)
    -- We should get up to 5 distinct chunk_ids in the 700-704 range
    -- among the top hits.
    T.gt(#collapsed, 0)
end)

T.test("upsert rejects wrong-dim vectors", function()
    T.err(function()
        hype.upsert(h, 9999, 1, "x", { 0.1, 0.2 })
    end, "expected " .. EMBED_DIM)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
