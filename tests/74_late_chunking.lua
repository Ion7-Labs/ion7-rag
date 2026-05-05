#!/usr/bin/env luajit
--- @module tests.74_late_chunking
---
--- Late chunking on a real embedder. Requires ION7_EMBED_MODEL.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.chunk.late — late-chunking embedder")

H.require_lsqlite3(T)
H.require_embed_model(T)
local ion7 = H.require_backend(T)

local llm       = require "ion7.llm"
local recursive = require "ion7.rag.chunk.recursive"
local late      = require "ion7.rag.chunk.late"
local rag_embed = require "ion7.rag.embed"

-- ── Setup ──────────────────────────────────────────────────────────────

local model = ion7.Model.load(H.embed_model_path(),
    { n_gpu_layers = H.gpu_layers() })

-- Pooled context for the "naive per-chunk" baseline encode.
local pooled_embedder = llm.Embed.new(model, {
    n_ctx     = 1024,
    pooling   = "last",
    n_threads = 4,
})
local probe = pooled_embedder:encode("probe")
local DIM = #probe

-- Token-level context (pooling = none) for late chunking. Needs to
-- hold the FULL document we're going to encode in one decode.
local ctx_late = model:embedding_context({
    n_ctx     = 4096,
    pooling   = "none",
    n_threads = 4,
})
local vocab = model:vocab()

local function cleanup()
    pooled_embedder:free()
    ctx_late:free()
    ion7.shutdown()
end

-- ── Test corpus ────────────────────────────────────────────────────────

local DOC = [[
# Annual Supply Agreement

## 1. Parties

Acme Industries SARL, registered in Lyon under SIRET 12345678900012,
and Beta Logistics SA, registered in Marseille under SIRET 98765432100021.

## 2. Pricing

Standard volume discount applies as follows : 3 % on the first tier,
5 % on the second tier, 7 % on the third tier, and 10 % beyond that.
Payment is due 60 days after invoice.

## 3. Delivery

Standard orders ship within 7 business days from the receipt of a
written order. Express orders ship the next business day.
]]

local doc = { text = DOC, sections = {} }

local function count_tokens(s) return math.ceil(#s / 4) end

-- Manually-chunked version so we can exercise both pipelines on the
-- exact same chunk boundaries.
local CHUNKS = recursive.chunk(doc, {
    count_tokens   = count_tokens,
    target_tokens  = 80,
    overlap_tokens = 0,
    min_tokens     = 20,
})

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("ctx exposes the new per-token embedding methods", function()
    T.is_type(ctx_late.embedding_token,     "function")
    T.is_type(ctx_late.embedding_token_ptr, "function")
    T.is_type(ctx_late.embeddings_all_ptr,  "function")
end)

T.test("late.encode returns one vector per chunk, of correct dim", function()
    local vecs = late.encode({
        ctx      = ctx_late,
        vocab    = vocab,
        doc_text = DOC,
        chunks   = CHUNKS,
        dim      = DIM,
    })
    T.eq(#vecs, #CHUNKS)
    for i, v in ipairs(vecs) do
        T.eq(#v, DIM, "chunk " .. i .. " vector has wrong dim : " .. #v)
    end
end)

T.test("different chunks yield distinct late-chunked vectors", function()
    local vecs = late.encode({
        ctx = ctx_late, vocab = vocab,
        doc_text = DOC, chunks = CHUNKS, dim = DIM,
    })
    if #vecs >= 2 then
        local sim = rag_embed.cosine(vecs[1], vecs[2])
        T.ok(sim < 0.999, "first two chunk vectors are too similar : " .. sim)
    end
end)

T.test("late vector differs from the naive in-isolation pooled encode", function()
    -- Same chunk, two encoding strategies : late chunking mean-pools
    -- per-token embeddings from a full-doc decode, naive encode runs
    -- last-token pooling on the chunk in isolation. The two operations
    -- produce different geometries — cosine can be anywhere in [-1, 1)
    -- but it must not be identically 1 (that would mean late chunking
    -- is a no-op). We assert non-identity and a non-zero magnitude on
    -- the late vector itself.
    local late_vecs = late.encode({
        ctx = ctx_late, vocab = vocab,
        doc_text = DOC, chunks = CHUNKS, dim = DIM,
    })
    local naive_vec = pooled_embedder:encode(CHUNKS[1].raw_text)

    local sim = rag_embed.cosine(late_vecs[1], naive_vec)
    T.ok(sim < 0.999, "late and naive vectors should not be identical")

    -- Late vector must carry non-trivial signal (not all zeros).
    local sq = 0
    for i = 1, #late_vecs[1] do sq = sq + late_vecs[1][i]^2 end
    T.gt(sq, 0, "late vector is all-zero — token embeddings unreadable")
end)

T.test("late.encode handles an oversized doc with a clear error", function()
    local huge = string.rep(DOC, 200)  -- way past 4096 tokens
    T.err(function()
        late.encode({
            ctx = ctx_late, vocab = vocab,
            doc_text = huge, chunks = CHUNKS, dim = DIM,
        })
    end, "ctx n_ctx is")
end)

T.test("late.encode returns empty when given no chunks", function()
    local vecs = late.encode({
        ctx = ctx_late, vocab = vocab,
        doc_text = DOC, chunks = {}, dim = DIM,
    })
    T.eq(#vecs, 0)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
