--- @module ion7.rag.route
--- @author  ion7 / Ion7 Project Contributors
---
--- Adaptive query router. Decides whether a query needs to hit the
--- retrieval pipeline at all, and when it does, which complexity tier
--- it belongs to. Three labels :
---
---   "no_retrieve"   chit-chat, math, translation, code-only requests
---   "single_hop"    factoid answerable from one good chunk
---   "multi_hop"     synthesises across multiple chunks
---
--- A TF-IDF + per-class centroid classifier matches LLM-class routing
--- F1 on RAGRouter-Bench (arXiv:2604.03455) because the cue words
--- ("what is X", "translate this", "tell me a joke") that distinguish
--- routing tiers are lexical, not semantic. The TF-IDF route adds
--- <1 ms per query and is the registered default ; callers that
--- already pay an LLM hop can plug in a grammar-constrained classifier.

local M = {}

local _ROUTERS = {
    tfidf = "ion7.rag.route.tfidf",
}

--- Resolve a router module by name.
---
--- @param  name  string  Currently `"tfidf"`.
--- @return table         The router module.
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _ROUTERS[name]
    if not mod then
        error("ion7.rag.route : unknown router '" .. tostring(name) ..
              "' (registered : tfidf)", 2)
    end
    return require(mod)
end

return M
