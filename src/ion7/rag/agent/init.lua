--- @module ion7.rag.agent
--- @author  ion7 / Ion7 Project Contributors
---
--- Generation-time RAG control loops.
---
---   CRAG (Yan et al., 2024)      Corrective : evaluate retrieval,
---                                  reformulate the query on low
---                                  confidence, retry.
---   Self-RAG (Asai et al., 2023) Reflective : decide-to-retrieve,
---                                  per-hit relevance, post-answer
---                                  support grading. Implemented via
---                                  ion7-grammar reflection-token
---                                  schemas — no fine-tuned model
---                                  required.
---
--- Both agents wrap an `ion7.rag.Pipeline` and never mutate it. They
--- are thin orchestrators over the Pipeline's primitives, which lets
--- callers swap CRAG for Self-RAG (or for plain `Pipeline:ask`)
--- without touching the indexing or retrieval setup.

local M = {}

local _AGENTS = {
    crag     = "ion7.rag.agent.crag",
    self_rag = "ion7.rag.agent.self_rag",
}

--- Resolve an agent module by name.
---
--- @param  name  string  `"crag"` | `"self_rag"`.
--- @return table         The agent module.
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _AGENTS[name]
    if not mod then
        error("ion7.rag.agent : unknown agent '" .. tostring(name) ..
              "' (registered : crag, self_rag)", 2)
    end
    return require(mod)
end

return M
