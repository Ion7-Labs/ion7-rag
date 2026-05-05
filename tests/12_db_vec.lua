#!/usr/bin/env luajit
--- @module tests.12_db_vec
--- @author  ion7 / Ion7 Project Contributors
---
--- sqlite-vec CRUD + KNN. Uses synthetic vectors so this file does NOT
--- require an embedder model.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.db.vec — sqlite-vec CRUD + KNN")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db  = require "ion7.rag.db"
local vec = require "ion7.rag.db.vec"

-- Smaller dims keep test vectors readable while still exercising the two
-- tiers (binary truncation + full-precision).
local EMBED_DIM  = 64
local BINARY_DIM = 32

local cp, ip = H.tmp_db_pair("12")
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

-- ── Helpers : synthetic vector generation ──────────────────────────────

math.randomseed(42)

local function random_unit_vector(dim)
    local v = {}
    local sum_sq = 0
    for i = 1, dim do
        v[i] = math.random() * 2 - 1
        sum_sq = sum_sq + v[i] * v[i]
    end
    local n = math.sqrt(sum_sq)
    for i = 1, dim do v[i] = v[i] / n end
    return v
end

--- Make a vector that is `noise` units away from `seed` along a random
--- direction, then re-normalise. Lower noise → closer to seed.
local function perturb(seed, noise)
    local out = {}
    local sum_sq = 0
    for i = 1, #seed do
        out[i] = seed[i] + (math.random() * 2 - 1) * noise
        sum_sq = sum_sq + out[i] * out[i]
    end
    local n = math.sqrt(sum_sq)
    for i = 1, #seed do out[i] = out[i] / n end
    return out
end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("upsert + count round-trip", function()
    vec.upsert(h, 1, random_unit_vector(EMBED_DIM))
    vec.upsert(h, 2, random_unit_vector(EMBED_DIM))
    T.eq(vec.count(h), 2)
end)

T.test("upsert is idempotent (replace by chunk_id)", function()
    vec.upsert(h, 100, random_unit_vector(EMBED_DIM))
    vec.upsert(h, 100, random_unit_vector(EMBED_DIM))
    T.eq(vec.count(h), 3)  -- 1, 2, 100 — not 4.
end)

T.test("upsert rejects wrong-dimension vectors", function()
    T.err(function() vec.upsert(h, 999, { 0.1, 0.2 }) end, "expected " .. EMBED_DIM)
end)

T.test("delete removes the row", function()
    vec.delete(h, 100)
    T.eq(vec.count(h), 2)
end)

T.test("upsert_many wraps a single transaction", function()
    local rows = {}
    for i = 1, 50 do
        rows[#rows + 1] = { chunk_id = 1000 + i, embedding = random_unit_vector(EMBED_DIM) }
    end
    vec.upsert_many(h, rows)
    T.eq(vec.count(h), 2 + 50)
end)

T.test("knn_full ranks the seed nearest to itself", function()
    -- Plant a known seed vector at chunk_id = 9001 ; perturb others.
    local seed = random_unit_vector(EMBED_DIM)
    vec.upsert(h, 9001, seed)
    for i = 1, 20 do
        vec.upsert(h, 9100 + i, perturb(seed, 0.5 + math.random()))
    end

    local hits = vec.knn_full(h, seed, 5)
    T.eq(#hits, 5, "expected 5 neighbours, got " .. #hits)
    T.eq(hits[1].chunk_id, 9001, "self should be the closest neighbour")
    T.gte(hits[2].distance, hits[1].distance,
        "distances must be non-decreasing")
end)

T.test("knn_binary returns the seed and orders monotonically", function()
    local seed = random_unit_vector(EMBED_DIM)
    vec.upsert(h, 9501, seed)
    for i = 1, 20 do
        vec.upsert(h, 9600 + i, perturb(seed, 0.5 + math.random()))
    end

    local hits = vec.knn_binary(h, seed, 5)
    T.eq(#hits, 5)
    -- Binary self-distance under sign-quantization is 0 only when no dim
    -- straddles 0 ; with truncation to BINARY_DIM = 32 it is at least
    -- always the smallest value across our planted set.
    T.eq(hits[1].chunk_id, 9501, "self should still rank first under binary")
    for i = 2, #hits do
        T.gte(hits[i].distance, hits[i-1].distance,
              "binary distances must be non-decreasing at i=" .. i)
    end
end)

T.test("knn search sizes respect k", function()
    local q = random_unit_vector(EMBED_DIM)
    T.eq(#vec.knn_full(h, q, 3),    3)
    T.eq(#vec.knn_binary(h, q, 7),  7)
end)

T.test("knn rejects invalid k", function()
    local q = random_unit_vector(EMBED_DIM)
    T.err(function() vec.knn_full(h, q, 0)   end, "k must be > 0")
    T.err(function() vec.knn_binary(h, q, -1) end, "k must be > 0")
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
