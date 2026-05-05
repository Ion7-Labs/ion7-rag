#!/usr/bin/env luajit
--- @module tests.11_db_chunks
--- @author  ion7 / Ion7 Project Contributors
---
--- docs + chunks CRUD against chunks.db. Foreign-key cascade.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.db.chunks — docs + chunks CRUD")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db     = require "ion7.rag.db"
local chunks = require "ion7.rag.db.chunks"

-- ── Fixture : open one shared handle for the file-level tests ───────────

local cp, ip = H.tmp_db_pair("11")
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

-- ── Tests ───────────────────────────────────────────────────────────────

T.test("insert_doc returns a positive PK", function()
    local pk = chunks.insert_doc(h, {
        doc_id      = "doc-001",
        format      = "markdown",
        title       = "Hello world",
        source_uri  = "/tmp/hello.md",
        ingested_at = 1714000000,
        meta_json   = '{"lang":"en"}',
    })
    T.is_type(pk, "number")
    T.gt(pk, 0)
end)

T.test("insert_doc rejects missing required fields", function()
    T.err(function() chunks.insert_doc(h, { format = "text" }) end, "doc_id")
    T.err(function() chunks.insert_doc(h, { doc_id = "x"     }) end, "format")
end)

T.test("insert_doc enforces UNIQUE(doc_id)", function()
    chunks.insert_doc(h, { doc_id = "dup-id", format = "text" })
    T.err(function()
        chunks.insert_doc(h, { doc_id = "dup-id", format = "text" })
    end, nil)
end)

T.test("get_doc_by_id round-trips every field", function()
    local pk = chunks.insert_doc(h, {
        doc_id      = "doc-002",
        format      = "html",
        title       = "Title 2",
        source_uri  = "https://example.com/2",
        ingested_at = 1714000100,
        meta_json   = '{"k":"v"}',
    })
    local row = chunks.get_doc_by_id(h, "doc-002")
    T.eq(row.id,          pk)
    T.eq(row.doc_id,      "doc-002")
    T.eq(row.format,      "html")
    T.eq(row.title,       "Title 2")
    T.eq(row.source_uri,  "https://example.com/2")
    T.eq(row.ingested_at, 1714000100)
    T.eq(row.meta_json,   '{"k":"v"}')
end)

T.test("get_doc_by_id returns nil for unknown id", function()
    T.eq(chunks.get_doc_by_id(h, "nope-not-here"), nil)
end)

T.test("insert_chunk returns a positive PK", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-chunks-1", format = "text" })
    local pk = chunks.insert_chunk(h, {
        doc_pk     = doc_pk,
        section    = "Intro",
        char_start = 0,
        char_end   = 17,
        n_tokens   = 5,
        raw_text   = "Hello, ion7-rag.",
    })
    T.is_type(pk, "number")
    T.gt(pk, 0)
end)

T.test("insert_chunk rejects missing required fields", function()
    T.err(function() chunks.insert_chunk(h, { doc_pk = 1, char_start = 0, char_end = 1 }) end, "raw_text")
    T.err(function() chunks.insert_chunk(h, { doc_pk = 1, raw_text = "x", char_end = 1 }) end, "char_start")
    T.err(function() chunks.insert_chunk(h, { doc_pk = 1, raw_text = "x", char_start = 0 }) end, "char_end")
end)

T.test("insert_chunks bulk-inserts in one transaction", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-chunks-bulk", format = "text" })
    local rows = {}
    for i = 0, 9 do
        rows[#rows + 1] = {
            doc_pk = doc_pk, section = "S" .. i,
            char_start = i * 100, char_end = i * 100 + 50,
            n_tokens = 12, raw_text = "chunk text " .. i,
        }
    end
    local pks = chunks.insert_chunks(h, rows)
    T.eq(#pks, 10)
    for i, pk in ipairs(pks) do T.gt(pk, 0, "pk[" .. i .. "] not positive") end
    T.eq(chunks.count_chunks(h, doc_pk), 10)
end)

T.test("get_chunk round-trips every field", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-chunks-2", format = "text" })
    local pk = chunks.insert_chunk(h, {
        doc_pk          = doc_pk,
        section         = "X > Y",
        char_start      = 100,
        char_end        = 250,
        n_tokens        = 30,
        raw_text        = "the quick brown fox",
        contextual_text = "About foxes : the quick brown fox",
        meta_json       = '{"who":"animals"}',
    })
    local c = chunks.get_chunk(h, pk)
    T.eq(c.id,              pk)
    T.eq(c.doc_pk,          doc_pk)
    T.eq(c.section,         "X > Y")
    T.eq(c.char_start,      100)
    T.eq(c.char_end,        250)
    T.eq(c.n_tokens,        30)
    T.eq(c.raw_text,        "the quick brown fox")
    T.eq(c.contextual_text, "About foxes : the quick brown fox")
    T.eq(c.meta_json,       '{"who":"animals"}')
end)

T.test("get_chunks preserves input order", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-chunks-order", format = "text" })
    local pks = chunks.insert_chunks(h, {
        { doc_pk = doc_pk, char_start = 0,   char_end = 10, raw_text = "AAA" },
        { doc_pk = doc_pk, char_start = 10,  char_end = 20, raw_text = "BBB" },
        { doc_pk = doc_pk, char_start = 20,  char_end = 30, raw_text = "CCC" },
    })
    -- Request in reverse + middle.
    local rows = chunks.get_chunks(h, { pks[3], pks[1], pks[2] })
    T.eq(rows[1].raw_text, "CCC")
    T.eq(rows[2].raw_text, "AAA")
    T.eq(rows[3].raw_text, "BBB")
end)

T.test("delete_doc cascades to chunks", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-cascade", format = "text" })
    chunks.insert_chunks(h, {
        { doc_pk = doc_pk, char_start = 0, char_end = 10, raw_text = "x" },
        { doc_pk = doc_pk, char_start = 10, char_end = 20, raw_text = "y" },
    })
    T.eq(chunks.count_chunks(h, doc_pk), 2)
    T.eq(chunks.delete_doc(h, doc_pk), 1)
    T.eq(chunks.count_chunks(h, doc_pk), 0,
        "FK ON DELETE CASCADE did not fire")
end)

T.test("iter_chunks_for_doc orders by char_start", function()
    local doc_pk = chunks.insert_doc(h, { doc_id = "for-iter", format = "text" })
    chunks.insert_chunks(h, {
        { doc_pk = doc_pk, char_start = 200, char_end = 210, raw_text = "Z" },
        { doc_pk = doc_pk, char_start = 100, char_end = 110, raw_text = "M" },
        { doc_pk = doc_pk, char_start = 0,   char_end = 10,  raw_text = "A" },
    })
    local seen = {}
    for c in chunks.iter_chunks_for_doc(h, doc_pk) do
        seen[#seen + 1] = c.raw_text
    end
    T.eq(table.concat(seen, ","), "A,M,Z")
end)

T.test("count_chunks total without doc filter", function()
    local n = chunks.count_chunks(h)
    T.gt(n, 0)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
