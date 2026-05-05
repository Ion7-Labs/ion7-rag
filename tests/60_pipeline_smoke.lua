#!/usr/bin/env luajit
--- @module tests.60_pipeline_smoke
---
--- End-to-end smoke test for `rag.Pipeline`. Loads an embedder + a
--- chat model, configures Enricher + Answerer + Pipeline, ingests a
--- tiny multi-topic corpus, retrieves, and asks one question that
--- can only be answered from one of the docs.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.Pipeline — end-to-end smoke")

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

-- ── Model loading ──────────────────────────────────────────────────────

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

local ctx_enrich = model_chat:context({
    n_ctx = 8192, n_seq_max = 1, n_threads = 4,
})
local ctx_answer = model_chat:context({
    n_ctx = 4096, n_seq_max = 1, n_threads = 4,
})

local enricher = rag.context.Enricher.new({
    ctx           = ctx_enrich,
    vocab         = model_chat:vocab(),
    max_tokens    = 64,
    max_doc_chars = 4000,
})

local cm_answer, engine_answer = llm.pipeline(ctx_answer, model_chat:vocab(), {
    headroom = 256,
})

-- ── Database + Pipeline ────────────────────────────────────────────────

local cp, ip = H.tmp_db_pair("60")
local h = rag.db.open({
    chunks_path     = cp,
    index_path      = ip,
    embed_dim       = EMBED_DIM,
    binary_dim      = BINARY_DIM,
    sqlite_vec_path = sqlite_vec_path,
})

local pipe = rag.Pipeline.new({
    handle    = h,
    embedder  = embedder,
    enricher  = enricher,
    answerer  = { engine = engine_answer, cm = cm_answer },
    chunker_opts = {
        target_tokens  = 256,
        overlap_tokens = 32,
        min_tokens     = 60,
    },
    retrieve_opts = { k_dense = 20, k_lex = 20, k_final = 5 },
    augment_top_k = 4,
})

local function cleanup()
    enricher:close()
    embedder:free()
    ctx_enrich:free()
    ctx_answer:free()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    ion7.shutdown()
end

-- ── Tiny multi-topic corpus ────────────────────────────────────────────

local DOCS = {
    {
        id = "delivery-policy",
        format = "markdown",
        text = [[
# Delivery and Shipping Policy

## Delivery Times

Standard orders ship within 7 business days of payment confirmation.
Express orders ship the next business day. International deliveries
may take an additional 5 to 10 business days depending on customs.

## Tracking

Every shipment includes a tracking code emailed at dispatch. Customers
can monitor delivery progress on the carrier's website using that code.

## Returns

Returns are accepted within 30 days of delivery for unopened items.
Damaged items must be reported within 48 hours.
]],
    },
    {
        id = "bread-recipe",
        format = "markdown",
        text = [[
# Country Sourdough Bread

## Ingredients

500 g of strong bread flour, 350 g of water, 100 g of mature sourdough
starter, and 10 g of fine sea salt. No yeast, no sugar.

## Method

Mix flour and water and let autolyse for 30 minutes. Add starter and
salt, then perform stretch-and-folds every 30 minutes for 2 hours.
Shape, cold-proof overnight in the fridge, and bake at 240 °C for
40 minutes inside a Dutch oven.
]],
    },
    {
        id = "bach-history",
        format = "markdown",
        text = [[
# Johann Sebastian Bach

## Life

Johann Sebastian Bach was born in Eisenach in 1685 and died in
Leipzig in 1750. He served as Kapellmeister at Köthen and as cantor
of the Thomasschule in Leipzig.

## Works

Bach composed the Brandenburg Concertos, the Goldberg Variations,
the Well-Tempered Clavier, and over 200 cantatas. His works form a
keystone of the Western classical canon.
]],
    },
}

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("ingest accepts an array of Docs and indexes them", function()
    local out = pipe:ingest(DOCS)
    T.eq(out.docs_ingested, 3)
    T.gte(out.chunks_indexed, 3, "expected at least one chunk per doc")
end)

T.test("contextual_text was set on every chunk by the enricher", function()
    local n_with_ctx
    for r in h:conn():nrows(
        "SELECT COUNT(*) AS n FROM chunks WHERE contextual_text IS NOT NULL")
    do n_with_ctx = r.n end
    T.gt(n_with_ctx, 0)
end)

T.test("retrieve(query) returns hydrated hits ordered by score", function()
    local hits = pipe:retrieve("How long does delivery take?")
    T.gt(#hits, 0)
    -- Every hit is hydrated.
    for _, hit in ipairs(hits) do
        T.is_type(hit.raw_text, "string")
        T.is_type(hit.doc_id,   "string")
    end
    -- Top hit should come from the delivery doc.
    T.eq(hits[1].doc_id, "delivery-policy",
        "expected top hit from delivery-policy, got " .. tostring(hits[1].doc_id))
end)

T.test("retrieve picks the right doc for a Bach query", function()
    local hits = pipe:retrieve("In what year was Bach born?")
    T.eq(hits[1].doc_id, "bach-history")
end)

T.test("retrieve picks the right doc for a recipe query", function()
    local hits = pipe:retrieve("How much flour for sourdough bread?")
    T.eq(hits[1].doc_id, "bread-recipe")
end)

T.test("ask returns a Response and the supporting hits", function()
    local response, hits = pipe:ask("How long do standard orders take to ship?",
                                     { max_tokens = 256 })
    T.is_type(response,         "table")
    T.is_type(response.content, "string")
    T.gt(#response.content, 10, "answer too short")
    T.gt(#hits, 0)
    -- The relevant doc must be in the hits ; the model is free to phrase
    -- the answer however it likes.
    local saw_delivery = false
    for _, hit in ipairs(hits) do
        if hit.doc_id == "delivery-policy" then saw_delivery = true ; break end
    end
    T.ok(saw_delivery, "delivery-policy should be in the retrieved hits")
end)

T.test("ask without an answerer raises clearly", function()
    local pipe_no_answer = rag.Pipeline.new({
        handle    = h,
        embedder  = embedder,
    })
    T.err(function() pipe_no_answer:ask("x") end, "no answerer was configured")
end)

T.test("mode = 'dense_only' skips FTS5 and still surfaces the right doc", function()
    local hits = pipe:retrieve("In what year was Bach born?",
                                { mode = "dense_only" })
    T.gt(#hits, 0)
    T.eq(hits[1].doc_id, "bach-history")
end)

T.test("mode = 'lex_only' skips the embedder forward pass", function()
    -- A keyword that uniquely belongs to the bread doc.
    local hits = pipe:retrieve("sourdough", { mode = "lex_only" })
    T.gt(#hits, 0)
    T.eq(hits[1].doc_id, "bread-recipe")
end)

T.test("mode rejects unknown values", function()
    T.err(function() pipe:retrieve("x", { mode = "fancy" }) end,
        "opts.mode must be 'hybrid'")
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
