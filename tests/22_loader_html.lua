#!/usr/bin/env luajit
--- @module tests.22_loader_html

local T = require "tests.framework"
require "tests.helpers"

T.suite("ion7.rag.loader.html — DOM walk via gumbo")

do
    local ok, _ = pcall(require, "gumbo")
    if not ok then
        T.skip("(this whole file)",
            "gumbo not installed — luarocks --local install gumbo")
        T.summary()
        os.exit(0)
    end
end

local html = require "ion7.rag.loader.html"

local SAMPLE = [[
<!DOCTYPE html>
<html>
<head><title>Doc Title</title></head>
<body>
<nav>Skip me</nav>
<header><h1>Top Heading</h1></header>
<main>
  <p>Paragraph one with some content.</p>
  <h2>Sub heading</h2>
  <p>Paragraph two under sub.</p>
  <h3>Deeper</h3>
  <p>Deep content.</p>
  <h2>Second sub</h2>
  <p>Body of second sub.</p>
  <script>var x = 1;</script>
  <style>body { color: red }</style>
</main>
<footer>Skip footer</footer>
</body>
</html>
]]

T.test("load() captures H1 as title", function()
    local doc = html.load(SAMPLE)
    T.eq(doc.title, "Top Heading")
end)

T.test("text drops script/style/nav/footer", function()
    local doc = html.load(SAMPLE)
    T.eq(doc.text:find("Skip me"),     nil, "nav content leaked")
    T.eq(doc.text:find("Skip footer"), nil, "footer content leaked")
    T.eq(doc.text:find("var x = 1"),   nil, "script content leaked")
    T.eq(doc.text:find("color: red"),  nil, "style content leaked")
end)

T.test("text retains body paragraphs", function()
    local doc = html.load(SAMPLE)
    T.contains(doc.text, "Paragraph one")
    T.contains(doc.text, "Paragraph two")
    T.contains(doc.text, "Deep content")
    T.contains(doc.text, "Body of second sub")
end)

T.test("section paths reflect heading hierarchy", function()
    local doc = html.load(SAMPLE)
    local paths = {}
    for _, s in ipairs(doc.sections) do paths[#paths + 1] = s.path end
    -- Expected: Top Heading, > Sub heading, > Sub heading > Deeper, > Second sub
    T.eq(paths[1], "Top Heading")
    T.eq(paths[2], "Top Heading > Sub heading")
    T.eq(paths[3], "Top Heading > Sub heading > Deeper")
    T.eq(paths[4], "Top Heading > Second sub")
end)

T.test("sections cover every byte of doc.text exactly once", function()
    local doc = html.load(SAMPLE)
    local prev_end = 0
    for i, s in ipairs(doc.sections) do
        T.eq(s.char_start, prev_end,
            string.format("section %d does not start where %d ended", i, i - 1))
        T.gt(s.char_end, s.char_start)
        prev_end = s.char_end
    end
    T.eq(prev_end, #doc.text)
end)

T.test("HTML with no headings yields one anonymous section", function()
    local doc = html.load("<html><body><p>Just one para.</p></body></html>")
    T.eq(#doc.sections, 1)
    T.eq(doc.sections[1].path,       "")
    T.eq(doc.sections[1].char_start, 0)
end)

T.test("falls back to <title> when no H1 is present", function()
    local doc = html.load(
        "<html><head><title>Fallback</title></head><body><p>Body.</p></body></html>")
    T.eq(doc.title, "Fallback")
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
