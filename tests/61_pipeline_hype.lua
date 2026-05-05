#!/usr/bin/env luajit
--- @module tests.61_pipeline_hype
---
--- End-to-end Pipeline test with HyPE enabled. Verifies the wiring :
---   - ingest produces hype rows alongside chunk rows
---   - retrieve picks them up automatically (use_hype = "auto")
---   - a question-shaped query still resolves to the right chunk

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.Pipeline + HyPE — end-to-end")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)
H.require_embed_model(T)
local context_model_path = H.context_model_path()
if not context_model_path then
    T.skip("(this whole file)",
        "set ION7_CONTEXT_MODEL or ION7_MODEL")
    T.summary()
    os.exit(0)
end

local ion7 = H.require_backend(T)
local llm  = require "ion7.llm"
local rag  = require "ion7.rag"

-- ── Models ─────────────────────────────────────────────────────────────

local model_embed = ion7.Model.load(H.embed_model_path(),
    { n_gpu_layers = H.gpu_layers() })
local embedder = llm.Embed.new(model_embed, {
    n_ctx     = 1024,
    pooling   = "last",
    n_threads = 4,
})
local probe = embedder:encode("probe")
local EMBED_DIM  = #probe
local BINARY_DIM = math.max(64, math.floor(EMBED_DIM / 8))

local model_chat = ion7.Model.load(context_model_path,
    { n_gpu_layers = H.gpu_layers() })

local ctx_enrich   = model_chat:context({ n_ctx = 8192, n_seq_max = 1, n_threads = 4 })
local ctx_hype     = model_chat:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })

local enricher = rag.context.Enricher.new({
    ctx           = ctx_enrich,
    vocab         = model_chat:vocab(),
    max_tokens    = 64,
    max_doc_chars = 4000,
})
local hype_gen = rag.hype.Generator.new({
    ctx           = ctx_hype,
    vocab         = model_chat:vocab(),
    n_questions   = 3,
    max_tokens    = 160,
    max_chunk_chars = 2000,
})

-- ── DB + Pipeline ──────────────────────────────────────────────────────

local cp, ip = H.tmp_db_pair("61")
local h = rag.db.open({
    chunks_path     = cp,
    index_path      = ip,
    embed_dim       = EMBED_DIM,
    binary_dim      = BINARY_DIM,
    sqlite_vec_path = sqlite_vec_path,
})

local pipe = rag.Pipeline.new({
    handle         = h,
    embedder       = embedder,
    enricher       = enricher,
    hype_generator = hype_gen,
    hype_n         = 3,
    chunker_opts   = { target_tokens = 256, overlap_tokens = 32, min_tokens = 60 },
    retrieve_opts  = { k_dense = 20, k_lex = 20, k_hype = 20, k_final = 5 },
})

local function cleanup()
    enricher:close()
    hype_gen:close()
    embedder:free()
    ctx_enrich:free()
    ctx_hype:free()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    ion7.shutdown()
end

-- ── Corpus ─────────────────────────────────────────────────────────────

local DOCS = {
    {
        id = "delivery-policy",
        format = "markdown",
        text = [[
# Delivery Policy

Standard orders ship within 7 business days. Express orders ship the
next business day. International orders take 5-10 extra days for customs.
]],
    },
    {
        id = "refund-policy",
        format = "markdown",
        text = [[
# Refund Policy

Refunds are issued within 14 days of return receipt. Damaged items
qualify for full refund regardless of return window.
]],
    },
}

-- ── Tests ──────────────────────────────────────────────────────────────

local ingest_result

T.test("ingest with hype_generator produces hype rows", function()
    ingest_result = pipe:ingest(DOCS)
    T.eq(ingest_result.docs_ingested, 2)
    T.gte(ingest_result.chunks_indexed, 2)
    T.gt(ingest_result.hype_indexed, 0,
        "expected at least one HyPE row to be indexed")
end)

T.test("hype_indexed roughly tracks chunks_indexed * hype_n", function()
    -- The model may produce fewer than 3 questions on a small chunk
    -- (truncation, mis-format). We just check we got ≥ chunks (every
    -- chunk produced at least one question).
    T.gte(ingest_result.hype_indexed, ingest_result.chunks_indexed,
        "expected hype_indexed ≥ chunks_indexed")
end)

T.test("idx.hype_vec stores chunk_id back-references", function()
    local n_with_chunk
    for r in h:conn():nrows(
        "SELECT COUNT(*) AS n FROM idx.hype_vec WHERE chunk_id IS NOT NULL")
    do n_with_chunk = r.n end
    T.eq(n_with_chunk, ingest_result.hype_indexed)
end)

T.test("retrieve auto-enables HyPE when index has hype rows", function()
    local hits = pipe:retrieve("How many days does delivery take?")
    T.gt(#hits, 0)
    T.eq(hits[1].doc_id, "delivery-policy",
        "expected delivery-policy on a delivery question")
end)

T.test("retrieve picks the right doc for a refund question", function()
    local hits = pipe:retrieve("Within how many days do you process refunds?")
    T.eq(hits[1].doc_id, "refund-policy")
end)

T.test("retrieve respects opts.use_hype = false", function()
    -- Force HyPE off, verify the call still works (it just skips the
    -- hype source). Quality may differ but shape stays sane.
    local hits = pipe:retrieve("How many days does delivery take?",
        { use_hype = false })
    T.gt(#hits, 0)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
