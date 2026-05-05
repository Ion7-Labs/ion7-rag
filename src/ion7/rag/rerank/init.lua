--- @module ion7.rag.rerank
--- @author  ion7 / Ion7 Project Contributors
---
--- Reranker registry. The registered `pointwise` reranker scores each
--- (query, document) pair via the yes/no logprob trick — the pattern
--- Qwen3-Reranker (June 2025) was designed for, and which any
--- chat-tuned model can run.
---
--- All rerankers expose the same surface :
---
---   reranker:score(query, document)         -> number  (higher = better)
---   reranker:rerank(handle, query, hits, k) -> Hit[]   sorted DESC by score

local M = {}

local _RERANKERS = {
    pointwise = "ion7.rag.rerank.pointwise",
}

--- Resolve a reranker module by name.
---
--- @param  name  string  Currently `"pointwise"`.
--- @return table         The reranker module.
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _RERANKERS[name]
    if not mod then
        error("ion7.rag.rerank : unknown reranker '" .. tostring(name) ..
              "' (registered : pointwise)", 2)
    end
    return require(mod)
end

return M
