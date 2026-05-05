--- @module ion7.rag
--- @author  ion7 / Ion7 Project Contributors
---
--- Ion7 RAG — Retrieval-Augmented Generation as a pure-Lua library.
---
--- The layer between ion7-core (embeddings, rerankers, generation),
--- ion7-llm (chat pipeline, multi-session pool, prefix cache),
--- ion7-grammar (constrained output) and a working RAG system :
--- ingestion, chunking, contextual retrieval, hybrid (dense + BM25)
--- search, fusion, reranking, agentic loops, evaluation.
---
--- Storage substrate : two SQLite files.
---   `chunks.db` carries canonical text, document metadata, provenance,
---     and citations.
---   `index.db`  carries sqlite-vec dense vectors and FTS5 BM25 ; it is
---     disposable and rebuildable from `chunks.db`.
---
--- Out of scope :
---   HTTP / SSE / WebSocket transports, server endpoints, CLI binaries.
---   Distributed vector stores, cloud-native ANN backends. ion7-rag is a
---   library, not a server.
---
--- Module loading is lazy : accessing a registered class or sub-
--- namespace triggers the require ; subsequent reads are direct table
--- lookups.

local rag = {}

-- ── Class registry (lazy) ─────────────────────────────────────────────────

local _CLASSES = {
    -- Hybrid retrieval glue. Exposes plain functions
    -- (`rag.retrieve.search(h, opts)`) ; rides the class registry only
    -- to benefit from the lazy-require mechanism.
    retrieve = "ion7.rag.retrieve",
    -- Embedder helpers built on top of an externally-instantiated
    -- `ion7.llm.Embed`.
    embed    = "ion7.rag.embed",
}

-- The Pipeline class is exposed at the top of the public API rather
-- than under a sub-namespace because it is the first surface most
-- consumers reach for.
local function _pipeline_resolver()
    return require("ion7.rag.pipeline").Pipeline
end

-- ── Sub-namespaces (lazy) ─────────────────────────────────────────────────
--
-- Each entry returns a table that itself lazy-loads its children. The
-- factory result is cached on the outer table after first access.

local function make_lazy(map)
    local t = {}
    return setmetatable(t, {
        __index = function(_, k)
            local p = map[k]
            if not p then return nil end
            local mod = require(p)
            rawset(t, k, mod)
            return mod
        end,
    })
end

local _NAMESPACES = {
    db = function()
        local Db = require "ion7.rag.db"
        return setmetatable({
            Handle = Db.Handle,
            open   = Db.open,
            chunks = require "ion7.rag.db.chunks",
            vec    = require "ion7.rag.db.vec",
            lex    = require "ion7.rag.db.lex",
            hype   = require "ion7.rag.db.hype",
            schema = require "ion7.rag.db.schema",
        }, { __call = function(_, ...) return Db.open(...) end })
    end,

    loader = function()
        local L = require "ion7.rag.loader"
        return setmetatable({
            from_string   = L.from_string,
            from_file     = L.from_file,
            for_format    = L.for_format,
            detect_format = L.detect_format,
            text          = require "ion7.rag.loader.text",
            markdown      = require "ion7.rag.loader.markdown",
            html          = require "ion7.rag.loader.html",
        }, { __call = function(_, ...) return L.from_string(...) end })
    end,

    chunk = function()
        local C = require "ion7.rag.chunk"
        return setmetatable({
            for_name  = C.for_name,
            chunk     = C.chunk,
            recursive = require "ion7.rag.chunk.recursive",
            late      = require "ion7.rag.chunk.late",
        }, { __call = function(_, ...) return C.chunk(...) end })
    end,

    fusion = function()
        local F = require "ion7.rag.fusion"
        return setmetatable({
            for_name = F.for_name,
            fuse     = F.fuse,
            rrf      = require "ion7.rag.fusion.rrf",
            dbsf     = require "ion7.rag.fusion.dbsf",
            cc       = require "ion7.rag.fusion.cc",
        }, { __call = function(_, ...) return F.fuse(...) end })
    end,

    rerank = function()
        local R = require "ion7.rag.rerank"
        return setmetatable({
            for_name  = R.for_name,
            Pointwise = require("ion7.rag.rerank.pointwise").Pointwise,
            pointwise = require "ion7.rag.rerank.pointwise",
        }, { __index = R })
    end,

    context = function()
        local C = require "ion7.rag.context"
        return setmetatable({
            Enricher = C.Enricher,
            prompts  = C.prompts,
        }, { __index = C })
    end,

    route = function()
        local R = require "ion7.rag.route"
        return setmetatable({
            for_name = R.for_name,
            tfidf    = require "ion7.rag.route.tfidf",
        }, { __index = R })
    end,

    hype = function()
        return require "ion7.rag.hype"
    end,

    agent = function()
        local A = require "ion7.rag.agent"
        return setmetatable({
            for_name = A.for_name,
            CRAG     = require("ion7.rag.agent.crag").CRAG,
            SelfRAG  = require("ion7.rag.agent.self_rag").SelfRAG,
            crag     = require "ion7.rag.agent.crag",
            self_rag = require "ion7.rag.agent.self_rag",
            prompts  = require "ion7.rag.agent.prompts",
        }, { __index = A })
    end,

    eval = function()
        local E = require "ion7.rag.eval"
        return setmetatable({
            for_name         = E.for_name,
            Faithfulness     = require("ion7.rag.eval.faithfulness").Faithfulness,
            ContextPrecision = require("ion7.rag.eval.context_precision").ContextPrecision,
            Lynx             = require("ion7.rag.eval.lynx").Lynx,
            faithfulness     = require "ion7.rag.eval.faithfulness",
            context_precision = require "ion7.rag.eval.context_precision",
            lynx             = require "ion7.rag.eval.lynx",
            prompts          = require "ion7.rag.eval.prompts",
        }, { __index = E })
    end,

    util = function() return make_lazy({
        log = "ion7.rag.util.log",
    }) end,
}

setmetatable(rag, {
    __index = function(t, k)
        if k == "Pipeline" then
            local cls = _pipeline_resolver()
            rawset(t, k, cls)
            return cls
        end
        local class_path = _CLASSES[k]
        if class_path then
            local mod = require(class_path)
            rawset(t, k, mod)
            return mod
        end
        local ns_factory = _NAMESPACES[k]
        if ns_factory then
            local ns = ns_factory()
            rawset(t, k, ns)
            return ns
        end
        return nil
    end,
})

-- ── Version + capability snapshot ────────────────────────────────────────

rag.VERSION = "0.1.0-alpha1"

--- Probe the ion7-rag runtime and report which sibling layers resolve
--- from the current `package.path`. Pure-Lua check ; does not open any
--- native library or model file.
---
--- @return table {
---     version          string  ion7-rag version (matches `rag.VERSION`).
---     has_core         bool    `ion7.core` reachable on package.path.
---     has_llm          bool    `ion7.llm` reachable.
---     has_grammar      bool    `ion7.grammar` reachable.
---     core_version     string?  `ion7.core.VERSION` if reachable.
---     llm_version      string?
---     grammar_version  string?
--- }
function rag.capabilities()
    local function probe(mod)
        local ok, m = pcall(require, mod)
        if not ok or type(m) ~= "table" then return false, nil end
        return true, m.VERSION or m._VERSION
    end

    local has_core,    core_v    = probe("ion7.core")
    local has_llm,     llm_v     = probe("ion7.llm")
    local has_grammar, grammar_v = probe("ion7.grammar")

    return {
        version         = rag.VERSION,
        has_core        = has_core,
        has_llm         = has_llm,
        has_grammar     = has_grammar,
        core_version    = core_v,
        llm_version     = llm_v,
        grammar_version = grammar_v,
    }
end

return rag
