#!/usr/bin/env luajit
--- @module tests.00_modules
--- @author  ion7 / Ion7 Project Contributors
---
--- Verify that every Lua module under `src/ion7/rag/` loads cleanly,
--- WITHOUT depending on a built `libllama.so`, the `ion7_bridge.so`,
--- or `lsqlite3` / `lpeg` / `cjson` C libraries.
---
--- Same idea as ion7-core / ion7-llm's `00_modules.lua` : we monkey-patch
--- `ffi.load` and pre-register the heavy native modules in
--- `package.preload` so requires resolve without actually opening any
--- shared object. Catches require typos, syntax errors, and missing FFI
--- symbols a happy-path smoke test might never hit.
---
--- Exit status :
---   0 — every module loaded.
---   1 — at least one require failed.

local T = require "tests.framework"
require "tests.helpers"

T.suite("Module load — every file under src/ion7/rag/ requires cleanly")

-- ── Stub the native surface ──────────────────────────────────────────────

local ffi = require "ffi"

local _stub = setmetatable({}, {
    __index = function() return function() end end,
    __call  = function() return nil end,
})

ffi.load = function() return _stub end

-- Pre-register native modules ion7-rag will start to require as later
-- phases ship. Stubbing them here keeps 00_modules.lua deterministic
-- regardless of whether `lsqlite3` / `lpeg` / `cjson` are installed.
package.preload["ion7.core.ffi.bridge"] = function() return _stub end
package.preload["lsqlite3"]              = function() return _stub end
package.preload["lpeg"]                  = function() return _stub end
package.preload["cjson"]                 = function() return _stub end
package.preload["cmark"]                 = function() return _stub end
package.preload["gumbo"]                 = function() return _stub end

-- ── Walk the src tree ────────────────────────────────────────────────────

local function find_lua(dir)
    local out = {}
    local p = io.popen("find " .. dir .. " -type f -name '*.lua' -not -path '*/.old/*'")
    if not p then return out end
    for line in p:lines() do out[#out + 1] = line end
    p:close()
    return out
end

local files = find_lua("./src/ion7/rag")
table.sort(files)

T.test("at least one module discovered (sanity check)", function()
    T.gt(#files, 0, "find returned no .lua files — wrong CWD?")
end)

-- ── Try to require each file ────────────────────────────────────────────

for _, file in ipairs(files) do
    local mod = file
        :gsub("^%./src/", "")
        :gsub("%.lua$",   "")
        :gsub("/",        ".")
        :gsub("%.init$",  "")

    T.test("require '" .. mod .. "'", function()
        local ok, err = pcall(require, mod)
        if not ok then error(tostring(err):sub(1, 400), 0) end
    end)
end

-- ── Façade walk ─────────────────────────────────────────────────────────
--
-- The lazy façade in init.lua exposes a class registry, sub-namespaces,
-- a VERSION string, and a `capabilities()` probe. The class registry is
-- empty in v0.1.0-alpha1 — only the util sub-namespace is wired today.
-- New entries land as the v1 phases ship ; this section is extended in
-- lockstep.

T.suite("Façade — sub-namespaces")

local rag = require "ion7.rag"

T.test("rag.util exposes log", function()
    T.is_type(rag.util,     "table")
    T.is_type(rag.util.log, "table")
    T.is_type(rag.util.log.set_level, "function")
    T.is_type(rag.util.log.info,      "function")
end)

T.test("rag.db exposes Handle/open/chunks/vec/lex/hype/schema", function()
    T.is_type(rag.db,         "table")
    T.is_type(rag.db.Handle,  "table")
    T.is_type(rag.db.open,    "function")
    T.is_type(rag.db.chunks,  "table")
    T.is_type(rag.db.vec,     "table")
    T.is_type(rag.db.lex,     "table")
    T.is_type(rag.db.hype,    "table")
    T.is_type(rag.db.schema,  "table")
    T.eq(getmetatable(rag.db).__call ~= nil, true,
        "rag.db namespace should be callable as rag.db(opts)")
end)

T.test("rag.loader exposes from_string/from_file/text/markdown/html", function()
    T.is_type(rag.loader,           "table")
    T.is_type(rag.loader.from_string, "function")
    T.is_type(rag.loader.from_file,   "function")
    T.is_type(rag.loader.detect_format, "function")
    T.is_type(rag.loader.text,      "table")
    T.is_type(rag.loader.markdown,  "table")
    T.is_type(rag.loader.html,      "table")
end)

T.test("rag.chunk exposes recursive + late", function()
    T.is_type(rag.chunk,           "table")
    T.is_type(rag.chunk.for_name,  "function")
    T.is_type(rag.chunk.chunk,     "function")
    T.is_type(rag.chunk.recursive, "table")
    T.is_type(rag.chunk.recursive.chunk, "function")
    T.is_type(rag.chunk.late,      "table")
    T.is_type(rag.chunk.late.encode, "function")
end)

T.test("rag.fusion exposes rrf/dbsf/cc + fuse", function()
    T.is_type(rag.fusion,           "table")
    T.is_type(rag.fusion.for_name,  "function")
    T.is_type(rag.fusion.fuse,      "function")
    T.is_type(rag.fusion.rrf,       "table")
    T.is_type(rag.fusion.dbsf,      "table")
    T.is_type(rag.fusion.cc,        "table")
    T.is_type(rag.fusion.rrf.fuse,  "function")
    T.is_type(rag.fusion.dbsf.fuse, "function")
    T.is_type(rag.fusion.cc.fuse,   "function")
end)

T.test("rag.retrieve exposes search", function()
    T.is_type(rag.retrieve,        "table")
    T.is_type(rag.retrieve.search, "function")
end)

T.test("rag.embed exposes encode_query/index_chunks/cosine", function()
    T.is_type(rag.embed,              "table")
    T.is_type(rag.embed.encode_query, "function")
    T.is_type(rag.embed.index_chunks, "function")
    T.is_type(rag.embed.cosine,       "function")
end)

T.test("rag.rerank exposes Pointwise + pointwise namespace", function()
    T.is_type(rag.rerank,           "table")
    T.is_type(rag.rerank.Pointwise, "table")
    T.is_type(rag.rerank.Pointwise.new, "function")
    T.is_type(rag.rerank.for_name,  "function")
end)

T.test("rag.context exposes Enricher + prompts", function()
    T.is_type(rag.context,             "table")
    T.is_type(rag.context.Enricher,    "table")
    T.is_type(rag.context.Enricher.new, "function")
    T.is_type(rag.context.prompts,     "table")
    T.is_type(rag.context.prompts.SYSTEM,      "string")
    T.is_type(rag.context.prompts.USER_FORMAT, "string")
end)

T.test("rag.Pipeline is the high-level orchestrator class", function()
    T.is_type(rag.Pipeline,     "table")
    T.is_type(rag.Pipeline.new, "function")
    T.is_type(rag.Pipeline.DEFAULTS, "table")
end)

T.test("rag.route exposes tfidf.Router", function()
    T.is_type(rag.route,             "table")
    T.is_type(rag.route.for_name,    "function")
    T.is_type(rag.route.tfidf,       "table")
    T.is_type(rag.route.tfidf.Router,     "table")
    T.is_type(rag.route.tfidf.Router.fit, "function")
end)

T.test("rag.hype exposes Generator + parse helper", function()
    T.is_type(rag.hype,           "table")
    T.is_type(rag.hype.Generator, "table")
    T.is_type(rag.hype.Generator.new, "function")
    T.is_type(rag.hype.SYSTEM,        "string")
    T.is_type(rag.hype._parse_questions, "function")
end)

T.test("rag.agent exposes CRAG + SelfRAG", function()
    T.is_type(rag.agent,          "table")
    T.is_type(rag.agent.CRAG,     "table")
    T.is_type(rag.agent.CRAG.new, "function")
    T.is_type(rag.agent.SelfRAG,  "table")
    T.is_type(rag.agent.SelfRAG.new, "function")
    T.is_type(rag.agent.for_name, "function")
    T.is_type(rag.agent.prompts,  "table")
end)

T.test("rag.eval exposes Faithfulness/ContextPrecision/Lynx", function()
    T.is_type(rag.eval,                   "table")
    T.is_type(rag.eval.Faithfulness,      "table")
    T.is_type(rag.eval.Faithfulness.new,  "function")
    T.is_type(rag.eval.ContextPrecision,  "table")
    T.is_type(rag.eval.ContextPrecision.new, "function")
    T.is_type(rag.eval.Lynx,              "table")
    T.is_type(rag.eval.Lynx.new,          "function")
    T.is_type(rag.eval.for_name,          "function")
    T.is_type(rag.eval.prompts,           "table")
end)

T.suite("Façade — version + capabilities")

T.test("rag.VERSION is a non-empty string", function()
    T.is_type(rag.VERSION, "string")
    T.gt(#rag.VERSION, 0)
end)

T.test("rag.capabilities() returns a probe table", function()
    T.is_type(rag.capabilities, "function")
    local caps = rag.capabilities()
    T.is_type(caps,         "table")
    T.is_type(caps.version, "string")
    T.eq(caps.version, rag.VERSION, "capabilities.version must match rag.VERSION")
    -- The has_* probes are booleans regardless of whether the sibling
    -- module is reachable on package.path — they MUST NOT throw.
    T.is_type(caps.has_core,    "boolean")
    T.is_type(caps.has_llm,     "boolean")
    T.is_type(caps.has_grammar, "boolean")
end)

T.suite("Façade — undefined keys")

T.test("unknown key on rag returns nil (lazy __index does not throw)", function()
    T.eq(rag.NopeNotARealClass, nil)
    T.eq(rag.also_not_real,     nil)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
