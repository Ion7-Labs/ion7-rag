#!/usr/bin/env luajit
--- @module tests.21_loader_md

local T = require "tests.framework"
require "tests.helpers"

T.suite("ion7.rag.loader.markdown — section detection via cmark")

-- Skip the whole file when cmark is not installed, mirroring the
-- skip-don't-fail pattern used elsewhere.
do
    local ok, _ = pcall(require, "cmark")
    if not ok then
        T.skip("(this whole file)",
            "cmark not installed — luarocks --local install cmark")
        T.summary()
        os.exit(0)
    end
end

local md = require "ion7.rag.loader.markdown"

local SAMPLE = [[
# Document Title

Intro paragraph mentioning ion7-rag.

## Section A

Body of A. Multiple sentences. Still A.

### Subsection A.1

Deep nested content.

## Section B

Body of B.
]]

T.test("load() captures the H1 title", function()
    local doc = md.load(SAMPLE)
    T.eq(doc.title, "Document Title")
end)

T.test("text round-trips verbatim", function()
    local doc = md.load(SAMPLE)
    T.eq(doc.text, SAMPLE)
end)

T.test("sections cover every byte exactly once", function()
    local doc = md.load(SAMPLE)
    -- Sections should be ordered, non-overlapping, contiguous.
    local prev_end = 0
    for i, s in ipairs(doc.sections) do
        T.eq(s.char_start, prev_end,
            string.format("section %d does not start where %d ended", i, i - 1))
        T.gt(s.char_end, s.char_start,
            "section " .. i .. " is empty")
        prev_end = s.char_end
    end
    T.eq(prev_end, #SAMPLE, "sections do not cover the full text")
end)

T.test("section paths reflect heading hierarchy", function()
    local doc = md.load(SAMPLE)
    local paths = {}
    for _, s in ipairs(doc.sections) do paths[#paths + 1] = s.path end
    -- Expected : Document Title, > Section A, > Section A > Subsection A.1, > Section B
    T.eq(paths[1], "Document Title")
    T.eq(paths[2], "Document Title > Section A")
    T.eq(paths[3], "Document Title > Section A > Subsection A.1")
    T.eq(paths[4], "Document Title > Section B")
end)

T.test("section text contains the corresponding heading line", function()
    local doc = md.load(SAMPLE)
    -- Section 2 = "Section A" — its slice must start with `## Section A`.
    local sec_a = doc.sections[2]
    local sliced = SAMPLE:sub(sec_a.char_start + 1, sec_a.char_end)
    T.contains(sliced, "## Section A")
    T.contains(sliced, "Body of A")
end)

T.test("prelude before first heading becomes its own anonymous section", function()
    local with_prelude = "Some prelude.\n\n# Title\n\nBody.\n"
    local doc = md.load(with_prelude)
    T.eq(doc.sections[1].path, "")
    T.contains(SAMPLE:sub(0, 0) .. with_prelude:sub(
        doc.sections[1].char_start + 1, doc.sections[1].char_end), "prelude")
    T.eq(doc.sections[2].path, "Title")
end)

T.test("document with no headings yields one anonymous section", function()
    local doc = md.load("Just a paragraph.\nNo headings here.")
    T.eq(#doc.sections, 1)
    T.eq(doc.sections[1].path, "")
    T.eq(doc.sections[1].char_start, 0)
end)

T.test("to_plain_text strips structure to readable text", function()
    local s = "## Title\n\n**bold** and *italic*.\n\n- a\n- b\n"
    local out = md.to_plain_text(s)
    T.contains(out, "Title")
    T.contains(out, "bold")
    T.contains(out, "italic")
    T.eq(out:find("%*%*"), nil, "bold markup survived")
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
