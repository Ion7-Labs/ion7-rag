#!/usr/bin/env luajit
--- @module tests.24_ingest_idantic
---
--- End-to-end ingestion smoke test on a slice of the Idantic synthetic
--- dataset. Walks 5 .md files through loader → chunker → db.chunks /
--- db.lex.upsert. Skips when ION7_RAG_CORPUS is not set.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag — ingestion smoke on Idantic dataset")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)
do
    local ok = pcall(require, "cmark")
    if not ok then
        T.skip("(this whole file)",
            "cmark not installed — luarocks --local install cmark")
        T.summary()
        os.exit(0)
    end
end

local corpus_root = H.corpus_path()
if not corpus_root then
    T.skip("(this whole file)",
        "ION7_RAG_CORPUS not set — export ION7_RAG_CORPUS=/path/to/Idantic/Dataset/output")
    T.summary()
    os.exit(0)
end

local db        = require "ion7.rag.db"
local chunks_db = require "ion7.rag.db.chunks"
local lex       = require "ion7.rag.db.lex"
local loader    = require "ion7.rag.loader"
local recursive = require "ion7.rag.chunk.recursive"
local md_loader = require "ion7.rag.loader.markdown"

-- ── Discover .md files under the corpus root ────────────────────────────

local function list_md(root, limit)
    local out = {}
    local p = io.popen("find '" .. root .. "' -type f -name '*.md' | sort")
    if not p then return out end
    for line in p:lines() do
        out[#out + 1] = line
        if #out >= limit then break end
    end
    p:close()
    return out
end

local md_files = list_md(corpus_root, 5)

if #md_files == 0 then
    T.skip("(this whole file)",
        "no .md files found under " .. corpus_root)
    T.summary()
    os.exit(0)
end

-- 4-chars-per-token approximation, same as 23_chunk_recursive — keeps
-- the test self-contained without needing an embedder model.
local function count_tokens(s) return math.ceil(#s / 4) end

-- ── Open a fresh DB pair ────────────────────────────────────────────────

local cp, ip = H.tmp_db_pair("24")
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

T.test("found .md files in the corpus", function()
    T.gt(#md_files, 0)
end)

local total_docs   = 0
local total_chunks = 0

T.test("loader.from_file opens each .md cleanly", function()
    for _, path in ipairs(md_files) do
        local doc = loader.from_file(path)
        T.eq(doc.format, "markdown")
        T.gt(#doc.text, 0, "empty doc.text for " .. path)
        T.gt(#doc.sections, 0,
            "no sections detected (heading-less doc?) : " .. path)
    end
end)

T.test("end-to-end ingestion : load + chunk + insert + index", function()
    for _, path in ipairs(md_files) do
        local doc = loader.from_file(path)

        -- Insert the doc row.
        local doc_pk = chunks_db.insert_doc(h, {
            doc_id     = doc.id,
            format     = doc.format,
            source_uri = doc.source_uri,
            title      = doc.title,
        })

        -- Chunk and bulk-insert.
        local cs = recursive.chunk(doc, {
            count_tokens   = count_tokens,
            target_tokens  = 256,
            overlap_tokens = 32,
            min_tokens     = 50,
        })
        T.gt(#cs, 0, "no chunks produced for " .. path)

        local rows = {}
        for _, c in ipairs(cs) do
            rows[#rows + 1] = {
                doc_pk     = doc_pk,
                section    = c.section,
                char_start = c.char_start,
                char_end   = c.char_end,
                n_tokens   = c.n_tokens,
                raw_text   = c.raw_text,
            }
        end
        local pks = chunks_db.insert_chunks(h, rows)
        T.eq(#pks, #cs)

        -- Lexical index : feed cmark-rendered plain text so FTS5 sees
        -- clean tokens, not markdown markup.
        local lex_rows = {}
        for i, c in ipairs(cs) do
            lex_rows[i] = {
                chunk_id = pks[i],
                text     = md_loader.to_plain_text(c.raw_text),
            }
        end
        lex.upsert_many(h, lex_rows)

        total_docs   = total_docs   + 1
        total_chunks = total_chunks + #cs
    end
end)

T.test("FTS5 BM25 returns hits on a known token from the corpus", function()
    -- The Idantic contracts mention "contrat" repeatedly ; tickets and
    -- emails carry French legal vocabulary. Try a generic French term
    -- present in any synthetic French business doc.
    for _, query in ipairs({ "contrat", "client", "facture", "livraison" }) do
        local hits = lex.search(h, query, 10)
        if #hits > 0 then
            T.gte(#hits, 1, "expected at least 1 hit for '" .. query .. "'")
            return
        end
    end
    error("no hits for any of the seeded French queries — corpus may not be French")
end)

T.test("counts add up across docs and chunks tables", function()
    T.eq(chunks_db.count_chunks(h), total_chunks,
        "chunks total mismatch")
    T.eq(lex.count(h), total_chunks,
        "FTS row count mismatch")
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
