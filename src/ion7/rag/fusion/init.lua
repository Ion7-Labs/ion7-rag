--- @module ion7.rag.fusion
--- @author  ion7 / Ion7 Project Contributors
---
--- Hybrid-retrieval fusion registry. Combines per-source ranked hit
--- lists (dense / lex / hype) into a single ranking.
---
--- All strategies share the same I/O shape :
---
---   hits_by_source = {                       opts = {
---       dense = { Hit, Hit, ... },               weights = { dense=4, lex=1 },
---       lex   = { Hit, Hit, ... },               k = 60,   -- RRF only
---       ...                                  }
---   }
---
---   Hit (input)  = { chunk_id, distance }    smaller distance = better
---   Hit (output) = { chunk_id, score    }    larger  score    = better
---
--- The Anthropic Contextual Retrieval paper recommends a 4:1
--- dense:lex weighting as a sane prior ; the optimal ratio depends on
--- corpus and embedder. CC and DBSF accept arbitrary weights and can
--- be tuned against labelled queries.

local M = {}

local _STRATEGIES = {
    rrf  = "ion7.rag.fusion.rrf",
    dbsf = "ion7.rag.fusion.dbsf",
    cc   = "ion7.rag.fusion.cc",
}

--- Resolve a fusion strategy module by name.
---
--- @param  name  string  `"rrf"` | `"dbsf"` | `"cc"`.
--- @return table         The strategy module (exposes `fuse`).
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _STRATEGIES[name]
    if not mod then
        error("ion7.rag.fusion : unknown strategy '" .. tostring(name) ..
              "' (registered : rrf, dbsf, cc)", 2)
    end
    return require(mod)
end

--- Pass-through to the named strategy's `fuse(hits_by_source, opts)`.
---
--- @param  name            string
--- @param  hits_by_source  table<string, Hit[]>
--- @param  opts            table?
--- @return Hit[]           Fused, descending by `score`.
function M.fuse(name, hits_by_source, opts)
    return M.for_name(name).fuse(hits_by_source, opts)
end

return M
