--- @module ion7.rag.eval.context_precision
--- @author  ion7 / Ion7 Project Contributors
---
--- RAGAs-style context precision, rank-weighted. For each retrieved
--- excerpt, the judge answers "is this excerpt relevant to the
--- question?" via the yes/no logprob trick. The metric weights early
--- hits more than late ones, in the spirit of nDCG :
---
---   score = (1 / |relevant|) * Σ_i  Precision@i * relevance_i
---
--- where `Precision@i` is the fraction of relevant items in positions
--- `1..i`. Range [0, 1] ; higher is better. A list of all-relevant
--- contexts in any order maxes out at 1.0 ; interleaving relevant and
--- irrelevant contexts pushes the score down sharply.

local prompts = require "ion7.rag.eval.prompts"

local M = {}

local function _coerce_context(c)
    if type(c) == "table" then
        return c.raw_text or c.text or c.content or ""
    end
    return tostring(c or "")
end

-- ── Class ───────────────────────────────────────────────────────────────

local ContextPrecision = {}
ContextPrecision.__index = ContextPrecision
M.ContextPrecision = ContextPrecision

--- Construct a ContextPrecision scorer.
---
--- @param  opts table {
---     judge_ctx       ion7.core.Context  REQUIRED.
---     judge_vocab     ion7.core.Vocab    REQUIRED.
---     judge_threshold number?  Default 0. `logp(yes) - logp(no)` above
---                  this counts as relevant.
--- }
--- @return ContextPrecision
--- @raise When `judge_ctx` or `judge_vocab` is missing.
function ContextPrecision.new(opts)
    opts = opts or {}
    local Pointwise = require("ion7.rag.rerank.pointwise").Pointwise

    local judge = Pointwise.new({
        ctx           = assert(opts.judge_ctx,   "ContextPrecision.new : opts.judge_ctx required"),
        vocab         = assert(opts.judge_vocab, "ContextPrecision.new : opts.judge_vocab required"),
        system        = prompts.CONTEXT_PRECISION_SYSTEM,
        user_format   = prompts.CONTEXT_PRECISION_USER_FORMAT,
        max_doc_chars = 3000,
    })

    return setmetatable({
        _judge     = judge,
        _threshold = opts.judge_threshold or 0,
    }, ContextPrecision)
end

--- Compute context precision for a question against its retrieved
--- contexts.
---
--- @param  question  string
--- @param  contexts  string[] | Hit[]  Retrieved contexts in rank
---                  order, best first. Plain strings and hit-shaped
---                  tables (`raw_text` / `text` / `content`) are both
---                  accepted.
--- @return number  Rank-weighted precision in [0, 1].
--- @return table   `{ scores : number[], relevant_mask : bool[],
---                  n_relevant : integer }`.
function ContextPrecision:score(question, contexts)
    if not contexts or #contexts == 0 then
        return 0, { scores = {}, relevant_mask = {}, n_relevant = 0 }
    end

    local scores        = {}
    local mask          = {}
    local n_relevant    = 0
    for i, c in ipairs(contexts) do
        local s = self._judge:score(question, _coerce_context(c))
        scores[i] = s
        if s > self._threshold then
            mask[i] = true
            n_relevant = n_relevant + 1
        else
            mask[i] = false
        end
    end

    if n_relevant == 0 then
        return 0, { scores = scores, relevant_mask = mask, n_relevant = 0 }
    end

    -- Σ_i (Precision@i × relevance_i) / |relevant|
    local cumulative_relevant = 0
    local total = 0
    for i = 1, #contexts do
        if mask[i] then
            cumulative_relevant = cumulative_relevant + 1
            total = total + (cumulative_relevant / i)
        end
    end

    return total / n_relevant, {
        scores        = scores,
        relevant_mask = mask,
        n_relevant    = n_relevant,
    }
end

return M
