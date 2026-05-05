#!/usr/bin/env luajit
--- @module tests.20_loader_text

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.loader.text — passthrough loader")

local loader = require "ion7.rag.loader"
local text   = require "ion7.rag.loader.text"

T.test("load() returns a Doc with text == input", function()
    local doc = text.load("Hello world.\nLine 2.")
    T.eq(doc.format, "text")
    T.eq(doc.text,   "Hello world.\nLine 2.")
    T.eq(#doc.sections, 0)
end)

T.test("load() honours opts (id, source_uri, title)", function()
    local doc = text.load("body", {
        id = "doc-X", source_uri = "/tmp/f.txt", title = "T",
    })
    T.eq(doc.id,         "doc-X")
    T.eq(doc.source_uri, "/tmp/f.txt")
    T.eq(doc.title,      "T")
end)

T.test("from_string dispatches by format", function()
    local doc = loader.from_string("hello", { format = "text" })
    T.eq(doc.format, "text")
    T.eq(doc.text,   "hello")
end)

T.test("from_string requires opts.format", function()
    T.err(function() loader.from_string("x") end, "format")
end)

T.test("from_file detects .txt extension", function()
    local path = H.tmpfile("ion7-rag-loader-text-fixture.txt")
    local f = io.open(path, "w") ; f:write("file body") ; f:close()
    local doc = loader.from_file(path)
    T.eq(doc.format,     "text")
    T.eq(doc.text,       "file body")
    T.eq(doc.source_uri, path)
    T.eq(doc.id,         path)
    H.try_remove(path)
end)

T.test("detect_format covers known extensions", function()
    T.eq(loader.detect_format("/x/y.txt"),      "text")
    T.eq(loader.detect_format("/x/y.md"),       "markdown")
    T.eq(loader.detect_format("/x/y.markdown"), "markdown")
    T.eq(loader.detect_format("/x/y.html"),     "html")
    T.eq(loader.detect_format("/x/y.htm"),      "html")
    T.eq(loader.detect_format("/x/y.unknown"),  nil)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
