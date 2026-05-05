#!/usr/bin/env luajit
--- @module tests.13_db_lex
--- @author  ion7 / Ion7 Project Contributors
---
--- FTS5 BM25 CRUD + search.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.db.lex — FTS5 BM25 CRUD + search")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db  = require "ion7.rag.db"
local lex = require "ion7.rag.db.lex"

local cp, ip = H.tmp_db_pair("13")
local h = db.open({
    chunks_path     = cp,
    index_path      = ip,
    embed_dim       = 1024,
    binary_dim      = 192,
    sqlite_vec_path = sqlite_vec_path,
})

local function cleanup()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("upsert + count round-trip", function()
    lex.upsert(h, 1, "the quick brown fox jumps over the lazy dog")
    lex.upsert(h, 2, "lorem ipsum dolor sit amet consectetur adipiscing")
    T.eq(lex.count(h), 2)
end)

T.test("upsert is idempotent (replace by rowid)", function()
    lex.upsert(h, 1, "the quick brown fox")  -- rewrite of rowid 1
    T.eq(lex.count(h), 2)
end)

T.test("delete removes a row", function()
    lex.delete(h, 2)
    T.eq(lex.count(h), 1)
end)

T.test("search returns hits where relevant", function()
    lex.upsert(h, 100, "machine learning models depend on training data")
    lex.upsert(h, 101, "the kitchen smells of fresh bread and cinnamon")
    lex.upsert(h, 102, "a learning rate that is too high causes divergence")
    lex.upsert(h, 103, "supervised learning beats unsupervised on these tasks")

    local hits = lex.search(h, "learning", 10)
    T.gte(#hits, 3, "expected at least 3 hits matching 'learning'")

    local seen = {}
    for _, hit in ipairs(hits) do seen[hit.chunk_id] = true end
    T.ok(seen[100], "rowid 100 missing from hits")
    T.ok(seen[102], "rowid 102 missing from hits")
    T.ok(seen[103], "rowid 103 missing from hits")
    T.ok(not seen[101], "non-matching rowid 101 leaked into hits")
end)

T.test("search distances are non-decreasing", function()
    local hits = lex.search(h, "learning", 10)
    for i = 2, #hits do
        T.gte(hits[i].distance, hits[i-1].distance,
              "distance must be non-decreasing at i=" .. i)
    end
end)

T.test("search respects k", function()
    local hits = lex.search(h, "learning", 2)
    T.eq(#hits, 2)
end)

T.test("search rejects empty / invalid args", function()
    T.err(function() lex.search(h, "",       5) end, "query")
    T.err(function() lex.search(h, "learning", 0) end, "k must be > 0")
end)

T.test("phrase queries work via FTS5 syntax", function()
    lex.upsert(h, 200, "foo bar baz")
    lex.upsert(h, 201, "bar baz foo")
    local hits = lex.search(h, '"foo bar"', 10)
    -- FTS5 phrase query should match only contexts where "foo bar" is
    -- adjacent in that order.
    local seen = {}
    for _, hit in ipairs(hits) do seen[hit.chunk_id] = true end
    T.ok(seen[200], "phrase hit on rowid 200 missing")
    T.ok(not seen[201], "phrase did not require ordered adjacency")
end)

T.test("multilingual unicode tokenization", function()
    lex.upsert(h, 300, "Le renard brun saute par-dessus le chien paresseux")
    lex.upsert(h, 301, "Der schnelle braune Fuchs springt über den faulen Hund")
    -- 'remove_diacritics 2' folds é/è to e ; 'paresseux' should match
    -- 'paresseux' regardless of accents in the query.
    local hits_fr = lex.search(h, "paresseux", 5)
    local seen_fr = false
    for _, h2 in ipairs(hits_fr) do if h2.chunk_id == 300 then seen_fr = true end end
    T.ok(seen_fr, "French token 'paresseux' did not match")
end)

T.test("upsert_many wraps a single transaction", function()
    local rows = {}
    for i = 1, 30 do
        rows[#rows + 1] = { chunk_id = 5000 + i, text = "bulk row " .. i }
    end
    lex.upsert_many(h, rows)
    local hits = lex.search(h, "bulk", 50)
    T.eq(#hits, 30)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
