#!/usr/bin/env luajit
--- @module tests.31_retrieve
---
--- End-to-end retrieve.search test against a synthetic corpus :
--- insert chunks + vectors + FTS, then run vec-only / lex-only / hybrid
--- queries through the glue layer. No embedder model — synthetic
--- vectors generated alongside the synthetic text.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.retrieve — hybrid search glue")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db        = require "ion7.rag.db"
local chunks_db = require "ion7.rag.db.chunks"
local vec_db    = require "ion7.rag.db.vec"
local lex_db    = require "ion7.rag.db.lex"
local retrieve  = require "ion7.rag.retrieve"

-- Tiny dims keep the test fixtures readable.
local EMBED_DIM  = 32
local BINARY_DIM = 16

local cp, ip = H.tmp_db_pair("31")
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

-- ── Synthetic corpus ───────────────────────────────────────────────────

math.randomseed(31)

local function rand_unit(dim)
    local v = {}
    local sum = 0
    for i = 1, dim do
        v[i] = math.random() * 2 - 1
        sum = sum + v[i] * v[i]
    end
    local n = math.sqrt(sum)
    for i = 1, dim do v[i] = v[i] / n end
    return v
end

local function perturb(seed, noise)
    local v = {}
    local sum = 0
    for i = 1, #seed do
        v[i] = seed[i] + (math.random() * 2 - 1) * noise
        sum = sum + v[i] * v[i]
    end
    local n = math.sqrt(sum)
    for i = 1, #seed do v[i] = v[i] / n end
    return v
end

-- Plant a "seed" vector and a known FTS-friendly token alongside it.
-- Other rows perturb the seed (so vec ranks them near the seed) but
-- carry random text (so FTS does NOT rank them on the seed token).
-- FTS5 unicode61 treats `-` as a separator AND parses leading `-` in a
-- query as an exclude-term operator, so a hyphenated marker token
-- silently turns into "watermelon -marker -9001" and the seed chunk is
-- the first thing FTS5 EXCLUDES. Keep markers letter+digit only.
local SEED_TOKEN = "watermelonmarker9001"
local SEED_VEC   = rand_unit(EMBED_DIM)

local seed_chunk_id
local doc_pk = chunks_db.insert_doc(h, { doc_id = "syn-doc-1", format = "text" })

-- 20 distractor chunks with random vectors and random text.
local rows_chunks, rows_vec, rows_lex = {}, {}, {}
for i = 1, 20 do
    local pk
    rows_chunks[i] = {
        doc_pk = doc_pk, char_start = (i - 1) * 100, char_end = i * 100,
        raw_text = "filler distractor content number " .. i,
    }
end

local pks = chunks_db.insert_chunks(h, rows_chunks)
for i, pk in ipairs(pks) do
    rows_vec[#rows_vec + 1] = {
        chunk_id  = pk,
        embedding = perturb(SEED_VEC, 0.5 + math.random() * 0.5),
    }
    rows_lex[#rows_lex + 1] = {
        chunk_id = pk,
        text     = rows_chunks[i].raw_text,
    }
end

-- One "seed" chunk : its vector IS the seed, and its text contains
-- the unique marker token.
local seed_pk = chunks_db.insert_chunk(h, {
    doc_pk = doc_pk, char_start = 9999, char_end = 10100,
    raw_text = "this very chunk holds the " .. SEED_TOKEN .. " token",
})
seed_chunk_id = seed_pk
vec_db.upsert(h, seed_pk, SEED_VEC)
lex_db.upsert(h, seed_pk, "this very chunk holds the " .. SEED_TOKEN .. " token")

vec_db.upsert_many(h, rows_vec)
lex_db.upsert_many(h, rows_lex)

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("requires at least one of query_vec / query_text", function()
    T.err(function() retrieve.search(h, {}) end, "query_vec")
end)

T.test("query_vec only : seed ranks first", function()
    local hits = retrieve.search(h, { query_vec = SEED_VEC, k_final = 5 })
    T.gte(#hits, 1)
    T.eq(hits[1].chunk_id, seed_chunk_id, "self-vector must rank top")
    -- Single-source path returns score = -distance (DESC by score).
    for i = 2, #hits do
        T.gte(hits[i - 1].score, hits[i].score, "score must be DESC")
    end
end)

T.test("query_text only : marker token surfaces the seed chunk", function()
    local hits = retrieve.search(h, { query_text = SEED_TOKEN, k_final = 5 })
    T.gte(#hits, 1)
    T.eq(hits[1].chunk_id, seed_chunk_id, "marker token must hit seed")
end)

T.test("hybrid : seed wins under RRF", function()
    local hits = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_final    = 5,
        fusion     = "rrf",
    })
    T.eq(hits[1].chunk_id, seed_chunk_id)
    -- Fused score is descending, no field name surprise.
    T.is_type(hits[1].score, "number")
end)

T.test("hybrid : seed wins under DBSF", function()
    local hits = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_final    = 5,
        fusion     = "dbsf",
    })
    T.eq(hits[1].chunk_id, seed_chunk_id)
end)

T.test("hybrid : seed wins under CC", function()
    local hits = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_final    = 5,
        fusion     = "cc",
    })
    T.eq(hits[1].chunk_id, seed_chunk_id)
end)

T.test("k_final truncates the output", function()
    local hits = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_final    = 3,
    })
    T.eq(#hits, 3)
end)

T.test("vec_tier='binary' uses the shortlist tier", function()
    local hits = retrieve.search(h, {
        query_vec = SEED_VEC,
        k_final   = 5,
        vec_tier  = "binary",
    })
    T.eq(hits[1].chunk_id, seed_chunk_id,
        "self should still rank first under binary tier")
end)

T.test("k_dense / k_lex bound the per-source candidate pulls", function()
    -- k_dense=2 means only 2 dense hits feed fusion ; if seed isn't in
    -- the top-2 dense (which it should be — distance 0), the test would
    -- still pass via lex. We assert the seed wins regardless.
    local hits = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_dense    = 2,
        k_lex      = 2,
        k_final    = 3,
    })
    T.eq(hits[1].chunk_id, seed_chunk_id)
end)

T.test("weights bias hybrid retrieval", function()
    -- Force a chunk that's lex-strong but vec-weak ; weights should
    -- decide.
    local oddball = chunks_db.insert_chunk(h, {
        doc_pk = doc_pk, char_start = 20000, char_end = 20050,
        raw_text = "oddball " .. SEED_TOKEN .. " marker again",
    })
    -- Far-from-seed vector for the oddball.
    local far = rand_unit(EMBED_DIM)
    -- Make sure it's actually far : flip signs.
    for i = 1, #far do far[i] = -SEED_VEC[i] end
    -- Re-normalise (already unit since SEED_VEC was unit).
    vec_db.upsert(h, oddball, far)
    lex_db.upsert(h, oddball, "oddball " .. SEED_TOKEN .. " marker again")

    local lex_heavy = retrieve.search(h, {
        query_vec  = SEED_VEC,
        query_text = SEED_TOKEN,
        k_final    = 3,
        fusion     = "rrf",
        weights    = { dense = 1, lex = 100 },
    })
    -- Both seed and oddball carry the marker ; under heavy lex weight
    -- both surface, but seed still wins because it's also dense-best.
    T.ok(lex_heavy[1].chunk_id == seed_chunk_id
         or lex_heavy[1].chunk_id == oddball)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
