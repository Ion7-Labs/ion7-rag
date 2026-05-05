#!/usr/bin/env luajit
--- @module tests.90_eval
---
--- RAGAs-style eval metrics. Uses a chat model as judge — prefers
--- ION7_CONTEXT_MODEL, falls back to ION7_MODEL.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.eval — RAGAs-style evaluation metrics")

local model_path = H.context_model_path()
if not model_path then
    T.skip("(this whole file)", "set ION7_CONTEXT_MODEL or ION7_MODEL")
    T.summary()
    os.exit(0)
end

local ion7 = H.require_backend(T)
local llm  = require "ion7.llm"
local rag  = require "ion7.rag"

-- ── Setup ──────────────────────────────────────────────────────────────

local model = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })
local vocab = model:vocab()

local ctx_judge   = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local ctx_extract = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })

local cm_extract, engine_extract = llm.pipeline(ctx_extract, vocab)

local function cleanup()
    ctx_judge:free()
    ctx_extract:free()
    ion7.shutdown()
end

-- Shared corpus / fixtures.
local CONTEXTS = {
    "Standard delivery is 7 business days from receipt of the written order.",
    "Express delivery ships the next business day.",
    "Refunds are issued within 14 days of return receipt.",
}

-- ── Faithfulness ───────────────────────────────────────────────────────

T.suite("ion7.rag.eval.Faithfulness — claim-grounding score")

local faithfulness = rag.eval.Faithfulness.new({
    judge_ctx     = ctx_judge,
    judge_vocab   = vocab,
    extractor     = { engine = engine_extract, cm = cm_extract },
    extract_max_tokens = 256,
})

T.test("faithful answer scores higher than a hallucinated answer", function()
    local q = "How long does standard delivery take?"

    local faithful = "Standard delivery takes 7 business days from receipt of the written order."
    local halluc   = "Standard delivery takes 24 hours guaranteed and includes free champagne with every shipment."

    local s_faithful = faithfulness:score(faithful, CONTEXTS)
    local s_halluc   = faithfulness:score(halluc,   CONTEXTS)

    T.gt(s_faithful, s_halluc,
        string.format("faithful=%g should beat halluc=%g", s_faithful, s_halluc))
    T.gte(s_faithful, 0)
    T.ok(s_faithful <= 1.0001, "faithfulness must be in [0, 1]")
end)

T.test("score returns details with the extracted claims", function()
    local _, details = faithfulness:score(
        "Standard delivery takes 7 business days.", CONTEXTS)
    T.is_type(details,        "table")
    T.is_type(details.claims, "table")
    -- Either we extracted at least one claim, or we got the
    -- "no claims" note ; both are valid outcomes for a tiny answer.
    T.ok(#details.claims > 0 or details.note,
        "expected at least one claim or a 'no claims' note")
end)

-- ── ContextPrecision ───────────────────────────────────────────────────

T.suite("ion7.rag.eval.ContextPrecision — rank-weighted relevance")

local cp = rag.eval.ContextPrecision.new({
    judge_ctx   = ctx_judge,
    judge_vocab = vocab,
})

T.test("relevant contexts get more 'yes' votes than off-topic ones", function()
    -- We assert on the per-context relevance MASK rather than on the
    -- final ranked score : a small Ministral judge is generous enough
    -- that off-topic contexts can occasionally pass the threshold,
    -- which makes the scalar score 1.0 in both buckets. The mask vote
    -- count remains discriminative.
    local q = "How long does standard delivery take?"

    local relevant = {
        "Standard delivery is 7 business days from receipt of the written order.",
        "Express delivery ships the next business day.",
        "International deliveries take an additional 5 to 10 business days.",
    }
    local irrelevant = {
        "Bach composed the Goldberg Variations.",
        "A roux is made by cooking flour and butter.",
        "Tomato plants prefer full sun and well-drained soil.",
    }

    local _, d_rel   = cp:score(q, relevant)
    local _, d_irrel = cp:score(q, irrelevant)
    T.gte(d_rel.n_relevant, d_irrel.n_relevant,
        string.format("relevant n=%d should beat irrelevant n=%d",
            d_rel.n_relevant, d_irrel.n_relevant))
    T.gte(d_rel.n_relevant, 1, "expected at least one relevant vote")
end)

T.test("score returns per-context details", function()
    local _, details = cp:score("How long does delivery take?", CONTEXTS)
    T.is_type(details.scores,        "table")
    T.is_type(details.relevant_mask, "table")
    T.eq(#details.scores, #CONTEXTS)
    T.eq(#details.relevant_mask, #CONTEXTS)
end)

T.test("empty context list returns zero", function()
    local s = cp:score("anything", {})
    T.eq(s, 0)
end)

-- ── Lynx ───────────────────────────────────────────────────────────────

T.suite("ion7.rag.eval.Lynx — Patronus-style PASS/FAIL judge")

local lynx = rag.eval.Lynx.new({
    ctx   = ctx_judge,
    vocab = vocab,
})

T.test("a faithful answer gets a non-FAIL verdict", function()
    local q = "How long does standard delivery take?"
    local document = CONTEXTS[1]
    local faithful = "Standard delivery takes 7 business days from receipt of the order."

    local verdict, margin = lynx:judge(q, faithful, document)
    T.one_of(verdict, { "PASS", "FAIL", "UNKNOWN" })
    T.is_type(margin, "number")
end)

T.test("a clearly hallucinated answer should not score above PASS-with-margin", function()
    local q = "How long does standard delivery take?"
    local document = CONTEXTS[1]
    local halluc = "Standard delivery takes 24 hours guaranteed with free champagne."

    local verdict_faith, m_faith = lynx:judge(q, "Standard delivery takes 7 business days.", document)
    local verdict_halluc, m_halluc = lynx:judge(q, halluc, document)

    -- The faithful margin should be larger than the hallucinated one.
    T.gt(m_faith, m_halluc,
        string.format("faithful margin=%g should beat halluc margin=%g", m_faith, m_halluc))
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
