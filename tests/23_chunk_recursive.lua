#!/usr/bin/env luajit
--- @module tests.23_chunk_recursive

local T = require "tests.framework"
require "tests.helpers"

T.suite("ion7.rag.chunk.recursive — recursive splitter")

local recursive = require "ion7.rag.chunk.recursive"

-- Synthetic token counter : ~4 chars per token. Roughly matches the
-- average token-to-char ratio of a Latin-script BPE tokenizer ; good
-- enough for chunker correctness tests, no model required.
local function count_tokens(s) return math.ceil(#s / 4) end

-- ── Helpers ─────────────────────────────────────────────────────────────

local function flat_doc(text)
    return { text = text, sections = {} }
end

local function repeated(text, n)
    local parts = {}
    for i = 1, n do parts[i] = text end
    return table.concat(parts)
end

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("opts.count_tokens is required", function()
    T.err(function() recursive.chunk(flat_doc("hello"), {}) end, "count_tokens")
end)

T.test("short text yields a single chunk", function()
    local doc = flat_doc("Hello world. Just a short doc.")
    local chunks = recursive.chunk(doc, {
        count_tokens = count_tokens,
        target_tokens = 100,
    })
    T.eq(#chunks, 1)
    T.eq(chunks[1].char_start, 0)
    T.eq(chunks[1].char_end,   #doc.text)
    T.eq(chunks[1].raw_text,   doc.text)
    T.eq(chunks[1].n_tokens, count_tokens(doc.text))
end)

T.test("long flat text splits to chunks under target", function()
    -- ~4000 chars ≈ 1000 tokens, target 100 tokens → at least 10 chunks.
    local body = repeated("The quick brown fox jumps over the lazy dog. ", 90)
    local doc  = flat_doc(body)
    local chunks = recursive.chunk(doc, {
        count_tokens = count_tokens,
        target_tokens = 100,
        overlap_tokens = 0,
        min_tokens     = 0,
    })
    T.gte(#chunks, 10)
    for _, c in ipairs(chunks) do
        T.gt(c.n_tokens, 0)
        -- Each chunk should be within ~1.5x target (slop for separator
        -- alignment) when overlap is off.
        T.ok(c.n_tokens <= 150,
            "chunk exceeded 1.5x target : " .. c.n_tokens)
    end
end)

T.test("chunks cover the full text when overlap = 0", function()
    local body = repeated("Sentence number X. ", 200)
    local doc  = flat_doc(body)
    local chunks = recursive.chunk(doc, {
        count_tokens   = count_tokens,
        target_tokens  = 100,
        overlap_tokens = 0,
        min_tokens     = 0,
    })
    -- With glue + no overlap, we should be able to reconstruct most of
    -- the text — separators between chunks may be lost (e.g. " " between
    -- words) but every NON-separator byte must be in exactly one chunk.
    -- Tighter property : the union of chunk spans covers a contiguous
    -- prefix of the text without gaps larger than 2 chars.
    local prev_end = 0
    for i, c in ipairs(chunks) do
        T.gte(c.char_start, prev_end - 2,
            "chunk " .. i .. " starts before previous end")
        T.gt(c.char_end, c.char_start)
        prev_end = c.char_end
    end
    T.gte(prev_end, #body - 2,
        "last chunk did not reach end of text")
end)

T.test("overlap shifts every chunk start back into the previous chunk", function()
    local body = repeated("word ", 400)  -- 2000 chars, ~500 tokens
    local doc  = flat_doc(body)
    local chunks = recursive.chunk(doc, {
        count_tokens   = count_tokens,
        target_tokens  = 50,
        overlap_tokens = 10,
        min_tokens     = 0,
    })
    T.gte(#chunks, 8, "expected several chunks")
    -- Chunk i+1's effective start should be inside chunk i's span.
    -- We stored char_start as the post-overlap offset, which means
    -- char_start of chunk i+1 should be < char_end of chunk i.
    for i = 2, #chunks do
        T.ok(chunks[i].char_start < chunks[i-1].char_end,
            "chunk " .. i .. " start (" .. chunks[i].char_start ..
            ") is past previous end (" .. chunks[i-1].char_end .. ")")
    end
end)

T.test("section-aware chunking respects section boundaries", function()
    -- Three sections with paths A, B, C ; chunker MUST NOT cross them.
    local doc = {
        text = "AAAA BBBB CCCC",
        sections = {
            { path = "A", char_start = 0,  char_end = 5  },  -- "AAAA "
            { path = "B", char_start = 5,  char_end = 10 },  -- "BBBB "
            { path = "C", char_start = 10, char_end = 14 },  -- "CCCC"
        },
    }
    local chunks = recursive.chunk(doc, {
        count_tokens   = count_tokens,
        target_tokens  = 100,
        overlap_tokens = 0,
        min_tokens     = 0,
    })
    T.eq(#chunks, 3, "expected one chunk per section")
    T.eq(chunks[1].section, "A")
    T.eq(chunks[2].section, "B")
    T.eq(chunks[3].section, "C")
    T.eq(chunks[1].char_end <= 5,  true)
    T.eq(chunks[2].char_start >= 5,  true)
    T.eq(chunks[2].char_end   <= 10, true)
    T.eq(chunks[3].char_start >= 10, true)
end)

T.test("min_tokens floor merges a tiny tail into the previous chunk", function()
    -- Build text where the natural last chunk is below the floor.
    local body = repeated("Word here. ", 50) .. "TINY"
    local doc  = flat_doc(body)
    local chunks = recursive.chunk(doc, {
        count_tokens   = count_tokens,
        target_tokens  = 60,
        overlap_tokens = 0,
        min_tokens     = 20,
    })
    -- Last chunk's text should include "TINY" and be at least min_tokens.
    local last = chunks[#chunks]
    T.contains(last.raw_text, "TINY")
    T.gte(last.n_tokens, 20, "trailing chunk did not reach min_tokens floor")
end)

T.test("chunk char_start/char_end map back to text correctly", function()
    local body = repeated("Sentence one. ", 30)
    local doc  = flat_doc(body)
    local chunks = recursive.chunk(doc, {
        count_tokens   = count_tokens,
        target_tokens  = 30,
        overlap_tokens = 0,
        min_tokens     = 0,
    })
    for i, c in ipairs(chunks) do
        T.eq(body:sub(c.char_start + 1, c.char_end), c.raw_text,
            "chunk " .. i .. " raw_text does not match offsets")
    end
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
