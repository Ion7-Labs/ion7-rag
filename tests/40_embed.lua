#!/usr/bin/env luajit
--- @module tests.40_embed
---
--- End-to-end embedder tests with a real GGUF. Requires
--- ION7_EMBED_MODEL (a Qwen3-Embedding GGUF or compatible).
---
--- The test corpus is small and synthetic so the assertions are about
--- the embedder's *qualitative* behaviour (semantic queries find the
--- right chunk), not absolute scores.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.embed — real embedder integration")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)
H.require_embed_model(T)
local ion7 = H.require_backend(T)

local llm       = require "ion7.llm"
local db        = require "ion7.rag.db"
local chunks_db = require "ion7.rag.db.chunks"
local rag_embed = require "ion7.rag.embed"
local retrieve  = require "ion7.rag.retrieve"

-- ── Model + embedder ───────────────────────────────────────────────────

local model_path = H.embed_model_path()
local model = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })
local embedder = llm.Embed.new(model, {
    n_ctx     = 1024,
    pooling   = "last",   -- Qwen3-Embedding family uses last-token pooling
    n_threads = 4,
})

-- Probe the embedder's actual output dimension by encoding a one-word
-- string. This makes the test work whether the user points at
-- Qwen3-Embedding-0.6B (1024-d), -4B (2560-d), or -8B (4096-d).
local probe_vec = embedder:encode("probe")
local EMBED_DIM = #probe_vec
local BINARY_DIM = math.max(64, math.floor(EMBED_DIM / 8))
print(string.format("[40_embed] embedder dim = %d, binary tier = %d",
    EMBED_DIM, BINARY_DIM))

local cp, ip = H.tmp_db_pair("40")
local h = db.open({
    chunks_path     = cp,
    index_path      = ip,
    embed_dim       = EMBED_DIM,
    binary_dim      = BINARY_DIM,
    sqlite_vec_path = sqlite_vec_path,
})

local function cleanup()
    embedder:free()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    ion7.shutdown()
end

-- ── Tiny semantic corpus ───────────────────────────────────────────────

local CORPUS = {
    { id = "rocket",   text = "Rocket engines achieve thrust by ejecting hot gases at high velocity." },
    { id = "cooking",  text = "A roux is made by cooking flour and butter together until golden." },
    { id = "music",    text = "Bach's Goldberg Variations are a set of thirty keyboard pieces." },
    { id = "geology",  text = "Sedimentary rocks form from layers of mineral particles compressed over time." },
    { id = "garden",   text = "Tomato plants prefer full sun and well-drained soil with consistent watering." },
}

local doc_pk = chunks_db.insert_doc(h, { doc_id = "embed-test", format = "text" })

local rows = {}
for _, item in ipairs(CORPUS) do
    rows[#rows + 1] = {
        doc_pk     = doc_pk,
        char_start = 0, char_end = #item.text,
        raw_text   = item.text,
        meta_json  = '{"id":"' .. item.id .. '"}',
    }
end
local pks = chunks_db.insert_chunks(h, rows)

local id_by_pk = {}
for i, item in ipairs(CORPUS) do id_by_pk[pks[i]] = item.id end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("embedder returns a vector of consistent dimension", function()
    local v1 = embedder:encode("hello world")
    local v2 = embedder:encode("a totally different topic")
    T.eq(#v1, EMBED_DIM)
    T.eq(#v2, EMBED_DIM)
end)

T.test("encode_query is a pass-through to embedder:encode", function()
    local v = rag_embed.encode_query(embedder, "probe")
    T.eq(#v, EMBED_DIM)
end)

T.test("cosine of a vector with itself is ~1", function()
    local v = embedder:encode("hello")
    local s = rag_embed.cosine(v, v)
    T.near(s, 1.0, 1e-3)
end)

T.test("index_chunks bulk-encodes and inserts vectors", function()
    local index_rows = {}
    for i, item in ipairs(CORPUS) do
        index_rows[i] = { chunk_id = pks[i], text = item.text }
    end
    local n = rag_embed.index_chunks(h, embedder, index_rows)
    T.eq(n, #CORPUS)

    local count
    for r in h:conn():nrows("SELECT COUNT(*) AS n FROM idx.chunks_vec") do
        count = r.n
    end
    T.eq(count, #CORPUS)
end)

T.test("vec-only retrieve finds the semantically-closest chunk", function()
    -- Each query targets one corpus item by topic.
    local probes = {
        { query = "How do rockets generate thrust?",                  expect = "rocket"  },
        { query = "Recipe basics: how to thicken a sauce?",            expect = "cooking" },
        { query = "Famous classical keyboard works by Bach",           expect = "music"   },
        { query = "How are rocks formed from layers over time?",       expect = "geology" },
        { query = "Tips for growing tomatoes in a backyard garden",    expect = "garden"  },
    }
    for _, p in ipairs(probes) do
        local qvec = embedder:encode(p.query)
        local hits = retrieve.search(h, { query_vec = qvec, k_final = 1 })
        T.eq(#hits, 1)
        T.eq(id_by_pk[hits[1].chunk_id], p.expect,
            string.format("query '%s' should find '%s', got '%s'",
                p.query, p.expect, id_by_pk[hits[1].chunk_id] or "?"))
    end
end)

T.test("binary tier shortlist also surfaces the right answer", function()
    local qvec = embedder:encode("How do rockets generate thrust?")
    local hits = retrieve.search(h, {
        query_vec = qvec,
        k_final   = 3,
        vec_tier  = "binary",
    })
    T.gte(#hits, 1)
    -- Binary is approximate ; we accept the right answer in the top 3.
    local top_ids = {}
    for _, hit in ipairs(hits) do top_ids[id_by_pk[hit.chunk_id]] = true end
    T.ok(top_ids["rocket"], "binary shortlist missed rocket in top-3")
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
