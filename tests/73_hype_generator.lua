#!/usr/bin/env luajit
--- @module tests.73_hype_generator
---
--- Standalone HyPE question Generator. Same skip-or-fall-back pattern
--- as the other model-driven tests : prefers ION7_CONTEXT_MODEL,
--- falls back to ION7_MODEL.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.hype.Generator — hypothetical question generation")

local model_path = H.context_model_path()
if not model_path then
    T.skip("(this whole file)",
        "set ION7_CONTEXT_MODEL or ION7_MODEL")
    T.summary()
    os.exit(0)
end

local ion7 = H.require_backend(T)
local hype = require "ion7.rag.hype"

local model = ion7.Model.load(model_path, { n_gpu_layers = H.gpu_layers() })
local ctx = model:context({ n_ctx = 4096, n_seq_max = 1, n_threads = 4 })
local vocab = model:vocab()

local gen = hype.Generator.new({
    ctx           = ctx,
    vocab         = vocab,
    n_questions   = 3,
    max_tokens    = 192,
    max_chunk_chars = 2000,
})

local function cleanup()
    gen:close()
    ctx:free()
    ion7.shutdown()
end

local CHUNK = [[
Standard delivery is 7 business days from receipt of the written order.
Express orders ship the next business day. International deliveries may
take an additional 5 to 10 business days depending on customs handling.
Every shipment includes a tracking code emailed at dispatch.
]]

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("generate returns an array of question strings", function()
    local qs = gen:generate(CHUNK, 3)
    T.is_type(qs, "table")
    T.gt(#qs, 0)
    for i, q in ipairs(qs) do
        T.is_type(q, "string", "question " .. i .. " is not a string")
        T.gt(#q, 5, "question " .. i .. " is too short")
    end
end)

T.test("questions are distinct", function()
    local qs = gen:generate(CHUNK, 3)
    local seen = {}
    for _, q in ipairs(qs) do
        T.eq(seen[q], nil, "duplicate question : " .. q)
        seen[q] = true
    end
end)

T.test("questions look like questions (often end with '?' for English)", function()
    -- Soft assertion : at least HALF the produced questions should
    -- contain a question mark or interrogative cue. Models occasionally
    -- produce statement-form prompts ; we don't fail on a single one.
    local qs = gen:generate(CHUNK, 5)
    T.gt(#qs, 0)
    local n_marked = 0
    for _, q in ipairs(qs) do
        if q:find("%?") or q:lower():match("^which") or q:lower():match("^what")
           or q:lower():match("^how") or q:lower():match("^when")
           or q:lower():match("^who") or q:lower():match("^where") then
            n_marked = n_marked + 1
        end
    end
    T.gte(n_marked, math.ceil(#qs / 2),
        "expected most questions to look interrogative ; got " ..
        n_marked .. "/" .. #qs)
end)

T.test("generate is robust to a giant chunk (truncation guard)", function()
    local huge = string.rep(CHUNK, 50)
    local qs = gen:generate(huge, 2)
    T.is_type(qs, "table")
end)

T.test("_parse_questions handles numbered, dashed, and asterisked lines", function()
    local sample = [[
1. What is the price ?
2) When does it ship ?
- Where is it located ?
* Why does it matter ?
   3.   Trimmed leading whitespace works
]]
    local out = hype._parse_questions(sample)
    T.gte(#out, 4)
end)

T.test("_parse_questions drops empties and duplicates", function()
    local sample = [[
1. Same line
2. Same line
1. Different line
]]
    local out = hype._parse_questions(sample)
    -- two distinct questions despite three numbered lines
    T.eq(#out, 2)
end)

cleanup()

local ok = T.summary()
os.exit(ok and 0 or 1)
