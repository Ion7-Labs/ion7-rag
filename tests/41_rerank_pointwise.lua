#!/usr/bin/env luajit
--- @module tests.41_rerank_pointwise
---
--- Pointwise LLM-as-judge reranker, on a chat-tuned model. The test
--- prefers ION7_RERANK_MODEL when set (a real Qwen3-Reranker GGUF) but
--- falls back to ION7_MODEL (any chat model) — the yes/no logprob
--- pattern works on both, just less well on a non-rerank-tuned model.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.rerank.pointwise — yes/no LLM-as-judge")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local rerank_model_path = H._env("ION7_RERANK_MODEL") or H._env("ION7_MODEL")
if not rerank_model_path then
    T.skip("(this whole file)",
        "set ION7_RERANK_MODEL or fall back to ION7_MODEL")
    T.summary()
    os.exit(0)
end

local ion7      = H.require_backend(T)
local db        = require "ion7.rag.db"
local chunks_db = require "ion7.rag.db.chunks"
local Pointwise = require("ion7.rag.rerank.pointwise").Pointwise

-- ── Setup ──────────────────────────────────────────────────────────────

local model = ion7.Model.load(rerank_model_path, { n_gpu_layers = H.gpu_layers() })
local ctx = model:context({ n_ctx = 2048, n_threads = 4 })
local vocab = model:vocab()

local rr = Pointwise.new({
    ctx           = ctx,
    vocab         = vocab,
    max_doc_chars = 1500,
})

-- A tiny db so :rerank can hydrate raw_text from chunk_ids.
local cp, ip = H.tmp_db_pair("41")
local h = db.open({
    chunks_path = cp, index_path = ip,
    embed_dim   = 1024, binary_dim = 192,
    sqlite_vec_path = sqlite_vec_path,
})

local function cleanup()
    ctx:free()
    h:close()
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    ion7.shutdown()
end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("score(query, relevant) > score(query, irrelevant)", function()
    local query    = "How do rocket engines generate thrust?"
    local good_doc = "Rocket engines work by ejecting mass at high velocity ; the reaction provides thrust per Newton's third law."
    local bad_doc  = "A roux is a mixture of flour and butter cooked together to thicken sauces in French cuisine."

    local s_good = rr:score(query, good_doc)
    local s_bad  = rr:score(query, bad_doc)

    T.is_type(s_good, "number")
    T.is_type(s_bad,  "number")
    T.gt(s_good, s_bad,
        string.format("relevant doc scored %.4f, irrelevant scored %.4f",
            s_good, s_bad))
end)

T.test("score(query, doc) is roughly stable across runs", function()
    local query = "What is photosynthesis?"
    local doc   = "Photosynthesis converts light energy into chemical energy stored in glucose."
    local s1 = rr:score(query, doc)
    local s2 = rr:score(query, doc)
    -- The model is deterministic on a clean KV ; identical decode →
    -- identical logprobs. Allow a small tolerance for any FFI / float
    -- nondeterminism on the GPU path.
    T.near(s1, s2, 0.05, "stability across two runs")
end)

T.test(":rerank reorders hits to put the relevant one first", function()
    -- Ingest 3 chunks ; a "rocket" chunk hidden in a sea of irrelevance.
    local doc_pk = chunks_db.insert_doc(h, { doc_id = "rerank-test", format = "text" })
    local pks = chunks_db.insert_chunks(h, {
        { doc_pk = doc_pk, char_start = 0, char_end = 1,
          raw_text = "A roux is made by cooking flour and butter to thicken sauces." },
        { doc_pk = doc_pk, char_start = 0, char_end = 1,
          raw_text = "Bach composed the Goldberg Variations for harpsichord." },
        { doc_pk = doc_pk, char_start = 0, char_end = 1,
          raw_text = "Rocket engines achieve thrust by ejecting hot gas at high velocity." },
    })

    -- Simulate a retriever that returned them in the WRONG order.
    local hits = {
        { chunk_id = pks[1], score = 0.9 }, -- "best" upstream score
        { chunk_id = pks[2], score = 0.8 },
        { chunk_id = pks[3], score = 0.7 }, -- the actually-relevant one
    }

    local reranked = rr:rerank(h, "How do rocket engines generate thrust?", hits)
    T.eq(#reranked, 3)
    T.eq(reranked[1].chunk_id, pks[3],
        "rocket chunk should top the rerank")
    -- Original score is preserved under prior_score.
    T.eq(reranked[1].prior_score, 0.7)
end)

T.test(":rerank truncates to k", function()
    local doc_pk = chunks_db.insert_doc(h, { doc_id = "rerank-trunc", format = "text" })
    local pks = chunks_db.insert_chunks(h, {
        { doc_pk = doc_pk, char_start = 0, char_end = 1, raw_text = "alpha topic." },
        { doc_pk = doc_pk, char_start = 0, char_end = 1, raw_text = "beta topic."  },
        { doc_pk = doc_pk, char_start = 0, char_end = 1, raw_text = "gamma topic." },
    })
    local hits = {}
    for i, pk in ipairs(pks) do hits[i] = { chunk_id = pk, score = 0 } end

    local out = rr:rerank(h, "alpha", hits, 2)
    T.eq(#out, 2)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
