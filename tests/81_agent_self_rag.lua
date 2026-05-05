#!/usr/bin/env luajit
--- @module tests.81_agent_self_rag
---
--- Self-RAG agent. Requires a chat model + an embedder + ion7-grammar
--- (to constrain the reflection-token JSON outputs).

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.agent.SelfRAG — reflection-token control loop")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)
H.require_embed_model(T)
local context_model_path = H.context_model_path()
if not context_model_path then
    T.skip("(this whole file)", "set ION7_CONTEXT_MODEL or ION7_MODEL")
    T.summary()
    os.exit(0)
end

-- Self-RAG depends on ion7-grammar for the reflection-token GBNF.
do
    local ok, _ = pcall(require, "ion7.grammar")
    if not ok then
        T.skip("(this whole file)",
            "ion7-grammar not reachable on package.path — set ION7_GRAMMAR_SRC")
        T.summary()
        os.exit(0)
    end
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
local ctx_answer = model_chat:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local cm_answer, engine_answer = llm.pipeline(ctx_answer, model_chat:vocab())

-- ── DB + Pipeline + SelfRAG agent ──────────────────────────────────────

local cp, ip = H.tmp_db_pair("81")
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
    answerer  = { engine = engine_answer, cm = cm_answer },
    chunker_opts  = { target_tokens = 200, overlap_tokens = 32, min_tokens = 50 },
    retrieve_opts = { k_dense = 10, k_lex = 10, k_final = 5 },
})

local agent = rag.agent.SelfRAG.new({
    pipeline   = pipe,
    grade_top_k = 3,
})

local function cleanup()
    embedder:free()
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
        text = "Standard orders ship within 7 business days. Express orders ship the next business day.",
    },
    {
        id = "refund", format = "text",
        text = "Refunds are issued within 14 days of return receipt.",
    },
}

pipe:ingest(DOCS)

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("SelfRAG.new requires a pipeline with an answerer or explicit judge", function()
    local pipe_no_answer = rag.Pipeline.new({ handle = h, embedder = embedder })
    T.err(function() rag.agent.SelfRAG.new({ pipeline = pipe_no_answer }) end,
        "judge engine pair is required")
end)

T.test("run on a retrieval-worthy question populates the decision log", function()
    local response, log = agent:run("How long does standard delivery take?")
    T.is_type(response.content, "string")
    T.is_type(log,             "table")
    T.is_type(log.decisions,   "table")
    -- Step 1 : retrieval decision.
    T.is_type(log.decisions.retrieve, "table")
    T.is_type(log.decisions.retrieve.retrieve, "boolean")
    T.is_type(log.decisions.retrieve.rationale, "string")
end)

T.test("when retrieve = true, relevance and support grades are recorded", function()
    local _, log = agent:run("How long does standard delivery take?")
    if log.decisions.retrieve and log.decisions.retrieve.retrieve then
        T.is_type(log.decisions.relevance, "table")
        T.gte(#log.decisions.relevance, 1, "expected at least one relevance grade")
        for i, d in ipairs(log.decisions.relevance) do
            T.is_type(d, "table", "relevance[" .. i .. "] not a table")
            T.is_type(d.relevant, "boolean")
        end
        T.is_type(log.decisions.support, "table")
        T.one_of(log.decisions.support.supported, { "full", "partial", "none" })
    end
end)

T.test("when retrieve = false, the agent answers directly", function()
    -- A pure conversational query should make the model vote no-retrieve.
    local response, log = agent:run("Hi, can you say hello back?")
    T.is_type(response.content, "string")
    T.gt(#response.content, 0)
    -- We don't fail if the model decides to retrieve anyway — small
    -- chat models can be over-eager — but we sanity-check the log.
    T.is_type(log.decisions.retrieve, "table")
    if log.decisions.retrieve.retrieve == false then
        T.eq(log.decisions.relevance, nil,
            "no relevance grades when not retrieving")
        T.eq(log.decisions.support, nil,
            "no support grade when not retrieving")
    end
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
