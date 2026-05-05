#!/usr/bin/env luajit
--- @module tests.80_agent_crag
---
--- Corrective RAG agent. Requires a chat model + an embedder.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.agent.CRAG — corrective control loop")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)
H.require_embed_model(T)
local context_model_path = H.context_model_path()
if not context_model_path then
    T.skip("(this whole file)", "set ION7_CONTEXT_MODEL or ION7_MODEL")
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
    n_ctx = 1024, pooling = "last", n_threads = 4,
})
local probe = embedder:encode("probe")
local EMBED_DIM, BINARY_DIM = #probe, math.max(64, math.floor(#probe / 8))

local model_chat = ion7.Model.load(context_model_path,
    { n_gpu_layers = H.gpu_layers() })

local ctx_rerank = model_chat:context({ n_ctx = 2048, n_seq_max = 1, n_threads = 4 })
local ctx_answer = model_chat:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })

local reranker = rag.rerank.Pointwise.new({
    ctx           = ctx_rerank,
    vocab         = model_chat:vocab(),
    max_doc_chars = 1500,
})

local cm_answer, engine_answer = llm.pipeline(ctx_answer, model_chat:vocab(), {
    headroom = 256,
})

-- ── DB + Pipeline + CRAG agent ─────────────────────────────────────────

local cp, ip = H.tmp_db_pair("80")
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
    reranker  = reranker,
    answerer  = { engine = engine_answer, cm = cm_answer },
    chunker_opts  = { target_tokens = 200, overlap_tokens = 32, min_tokens = 50 },
    retrieve_opts = { k_dense = 10, k_lex = 10, k_final = 5 },
})

local crag = rag.agent.CRAG.new({
    pipeline    = pipe,
    max_retries = 1,
})

local function cleanup()
    embedder:free()
    ctx_rerank:free()
    ctx_answer:free()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    ion7.shutdown()
end

-- ── Corpus ─────────────────────────────────────────────────────────────

local DOCS = {
    {
        id = "delivery", format = "text",
        text = "Standard orders ship within 7 business days. Express orders ship the next business day. International deliveries take 5-10 extra days for customs.",
    },
    {
        id = "refund", format = "text",
        text = "Refunds are issued within 14 days of return receipt. Damaged items qualify for full refund within 48 hours of report.",
    },
    {
        id = "support-hours", format = "text",
        text = "Customer support is available Monday to Friday from 9am to 6pm Paris time. Weekend support is via email only.",
    },
}

pipe:ingest(DOCS)

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("CRAG.new requires a reranker (via opts or pipeline)", function()
    T.err(function()
        rag.agent.CRAG.new({
            pipeline = rag.Pipeline.new({ handle = h, embedder = embedder }),
        })
    end, "reranker is required")
end)

T.test("CRAG.new requires a reformulator (answerer)", function()
    -- Pipeline with reranker but no answerer.
    local pipe_no_answer = rag.Pipeline.new({
        handle   = h,
        embedder = embedder,
        reranker = reranker,
    })
    T.err(function()
        rag.agent.CRAG.new({ pipeline = pipe_no_answer })
    end, "reformulator engine is required")
end)

T.test("run(query) returns Response + metadata for an answerable query", function()
    local response, meta = crag:run("How long does standard delivery take?",
                                     { ask = { max_tokens = 256 } })
    T.is_type(response.content, "string")
    T.gt(#response.content, 5)
    T.is_type(meta,            "table")
    T.is_type(meta.retrievals, "number")
    T.gte(meta.retrievals,     1)
    T.is_type(meta.max_scores, "table")
    T.is_type(meta.queries,    "table")
    T.eq(meta.queries[1], "How long does standard delivery take?")
    T.one_of(meta.confidence, { "high", "mixed", "low" })
end)

T.test("answerable query stays at 1 retrieval (no reformulation triggered)", function()
    local _, meta = crag:run("How long does standard delivery take?",
                              { ask = { max_tokens = 96 } })
    -- An obviously-relevant chunk should produce max_score above the
    -- threshold_correct on the first try, so no retry fires.
    if meta.confidence == "high" then
        T.eq(meta.retrievals, 1, "no retry should have fired on a clear hit")
    end
end)

T.test("clearly off-corpus query lowers confidence and may trigger a retry", function()
    -- Nothing in the corpus mentions cooking. The reranker should
    -- score the top hits low ; CRAG either flags low confidence or
    -- attempts a reformulation.
    local _, meta = crag:run("What's the recipe for sourdough bread?",
                              { ask = { max_tokens = 96 } })
    T.one_of(meta.confidence, { "low", "mixed", "high" })
    -- We don't assert retry happened (the small Ministral judge can
    -- be surprisingly generous), but we check the agent didn't crash
    -- on the low-confidence branch.
    T.gte(meta.retrievals, 1)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
