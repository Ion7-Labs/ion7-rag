#!/usr/bin/env luajit
--- @module tests.50_context_enricher
---
--- Anthropic Contextual Retrieval enricher. Requires a chat-tuned model
--- (ION7_CONTEXT_MODEL preferred, falling back to ION7_MODEL).

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.context.Enricher — Anthropic-style chunk contextualization")

local context_model_path = H.context_model_path()
if not context_model_path then
    T.skip("(this whole file)",
        "set ION7_CONTEXT_MODEL or ION7_MODEL")
    T.summary()
    os.exit(0)
end

local ion7  = H.require_backend(T)
local Enricher = require("ion7.rag.context").Enricher

-- ── Setup ──────────────────────────────────────────────────────────────

local model = ion7.Model.load(context_model_path, { n_gpu_layers = H.gpu_layers() })
local ctx   = model:context({
    n_ctx     = 8192,        -- room for a typical document + chunk + reply
    n_seq_max = 1,
    n_threads = 4,
})
local vocab = model:vocab()

local enricher = Enricher.new({
    ctx          = ctx,
    vocab        = vocab,
    max_tokens   = 96,
    max_doc_chars = 6000,
})

local function cleanup()
    enricher:close()
    ctx:free()
    ion7.shutdown()
end

-- ── Synthetic test doc ─────────────────────────────────────────────────

local FULL_DOC = [[
# Annual Supply Agreement CTR-2026-001

## 1. Parties

Acme Industries SARL, registered in Lyon under SIRET 12345678900012, and
Beta Logistics SA, registered in Marseille under SIRET 98765432100021,
hereinafter referred to as "the Supplier" and "the Customer".

## 2. Scope

The Supplier agrees to deliver electrical components according to the
catalogue annexed to this agreement. The agreement runs from
2026-01-01 to 2026-12-31, renewable yearly by tacit agreement unless
either party gives 30 days notice.

## 3. Pricing

Prices are fixed at the catalogue rates as of the signature date.
Volume discounts apply :
- 0-50 % of annual estimated volume : 3 % discount
- 51-80 % : 5 % discount
- 81-100 % : 7 % discount
- 100 %+ : 10 % discount

Payment terms : net 60 days from invoice date. Late payments incur
penalties at 1.5x the legal interest rate.

## 4. Delivery

Standard delivery is 7 business days from receipt of the written order.
The Supplier guarantees stock availability for the catalogue items.
]]

local CHUNK_PRICING = [[
Volume discounts apply :
- 0-50 % of annual estimated volume : 3 % discount
- 51-80 % : 5 % discount
- 81-100 % : 7 % discount
- 100 %+ : 10 % discount

Payment terms : net 60 days from invoice date. Late payments incur
penalties at 1.5x the legal interest rate.
]]

local CHUNK_DELIVERY = [[
Standard delivery is 7 business days from receipt of the written order.
The Supplier guarantees stock availability for the catalogue items.
]]

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("enrich_chunk produces a non-empty short context", function()
    local out = enricher:enrich_chunk(FULL_DOC, CHUNK_PRICING)
    T.is_type(out, "string")
    T.gt(#out, 20,  "context too short to be useful")
    T.ok(#out < 1500, "context too long ; max_tokens guard not effective : got " .. #out)
end)

T.test("enrich_chunk gives DIFFERENT contexts for different excerpts", function()
    local a = enricher:enrich_chunk(FULL_DOC, CHUNK_PRICING)
    local b = enricher:enrich_chunk(FULL_DOC, CHUNK_DELIVERY)
    T.neq(a, b, "two different chunks produced identical context")
end)

T.test("enrich_chunks mutates each chunk in place with contextual_text", function()
    local chunks = {
        { raw_text = CHUNK_PRICING },
        { raw_text = CHUNK_DELIVERY },
    }
    enricher:enrich_chunks(FULL_DOC, chunks)
    for i, c in ipairs(chunks) do
        T.is_type(c.contextual_text, "string", "chunk " .. i .. " missing contextual_text")
        T.gt(#c.contextual_text, #c.raw_text,
            "contextual_text should be longer than raw_text at chunk " .. i)
        T.contains(c.contextual_text, c.raw_text,
            "contextual_text must end with the raw chunk")
    end
end)

T.test("on_progress callback fires with (done, total)", function()
    local seen = {}
    local chunks = {
        { raw_text = "tiny chunk one." },
        { raw_text = "tiny chunk two." },
    }
    enricher:enrich_chunks(FULL_DOC, chunks, function(done, total)
        seen[#seen + 1] = { done, total }
    end)
    T.eq(#seen, 2)
    T.eq(seen[1][1], 1)
    T.eq(seen[1][2], 2)
    T.eq(seen[2][1], 2)
    T.eq(seen[2][2], 2)
end)

T.test("max_doc_chars truncates a giant document", function()
    -- Build a 50K-char document by repeating a known passage.
    local huge = string.rep(FULL_DOC .. "\n\n", 50)
    -- The contextualizer should still respond normally on a chunk from
    -- that huge document — the truncation guard prevents prompt blow-up.
    local out = enricher:enrich_chunk(huge, CHUNK_PRICING)
    T.is_type(out, "string")
    T.gt(#out, 0)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
