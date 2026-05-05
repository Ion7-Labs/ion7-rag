--- @module ion7.rag.eval
--- @author  ion7 / Ion7 Project Contributors
---
--- Reference-free RAG evaluation metrics, RAGAs-style (Es et al.,
--- arXiv:2309.15217). Each metric is a class instantiated against a
--- judge Context + Vocab ; scoring is one or more yes/no logprob
--- judgments routed through `ion7.rag.rerank.Pointwise`.
---
--- Registered metrics :
---
---   faithfulness       Ratio of answer claims grounded in the
---                      retrieved contexts.
---   context_precision  Rank-weighted relevance precision over the
---                      retrieved hit list.
---   lynx               Patronus-style PASS/FAIL hallucination judge.

local M = {}

local _METRICS = {
    faithfulness      = "ion7.rag.eval.faithfulness",
    context_precision = "ion7.rag.eval.context_precision",
    lynx              = "ion7.rag.eval.lynx",
}

--- Resolve a metric module by name.
---
--- @param  name  string  `"faithfulness"` | `"context_precision"` |
---                  `"lynx"`.
--- @return table         The metric module.
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _METRICS[name]
    if not mod then
        error("ion7.rag.eval : unknown metric '" .. tostring(name) ..
              "' (registered : faithfulness, context_precision, lynx)", 2)
    end
    return require(mod)
end

return M
