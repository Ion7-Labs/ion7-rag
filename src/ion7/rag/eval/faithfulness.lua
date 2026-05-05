--- @module ion7.rag.eval.faithfulness
--- @author  ion7 / Ion7 Project Contributors
---
--- RAGAs-style faithfulness score. Two-stage :
---
---   1. Extract atomic claims from the answer via the extractor
---      engine (free-form generation, line-based parse).
---   2. For each claim, ask the judge "is this claim supported by
---      the context?" via the yes/no logprob trick, routed through
---      `ion7.rag.rerank.Pointwise`.
---
--- score = n_supported / n_claims. Range [0, 1] ; higher is more
--- faithful.
---
--- The caller supplies a judge Context (for yes/no scoring) and an
--- extractor engine pair (for claim extraction). Both may share the
--- same Model but need SEPARATE Contexts : the judge does its own
--- `kv_clear` + `decode` per call and would clobber the extractor's
--- KV state otherwise.

local prompts = require "ion7.rag.eval.prompts"

local M = {}

-- ── Helpers ─────────────────────────────────────────────────────────────

local function _parse_claims(text)
    local seen, out = {}, {}
    for line in (text or ""):gmatch("[^\n]+") do
        local q = line:match("^%s*[%-%*]%s+(.+)$")
              or line:match("^%s*%d+[%.%)]%s+(.+)$")
        if q then
            q = q:gsub("^%s+", ""):gsub("%s+$", "")
            if #q > 4 and not seen[q] then
                seen[q] = true
                out[#out + 1] = q
            end
        end
    end
    return out
end

local function _join_contexts(contexts)
    if type(contexts) == "string" then return contexts end
    local parts = {}
    for i, c in ipairs(contexts) do
        if type(c) == "table" then
            parts[i] = c.raw_text or c.text or c.content or ""
        else
            parts[i] = tostring(c)
        end
    end
    return table.concat(parts, "\n\n")
end

local function _coerce_answer(answer)
    if type(answer) == "table" then return answer.content or "" end
    return tostring(answer or "")
end

-- ── Class ───────────────────────────────────────────────────────────────

local Faithfulness = {}
Faithfulness.__index = Faithfulness
M.Faithfulness = Faithfulness

--- Construct a Faithfulness scorer.
---
--- @param  opts table {
---     judge_ctx          ion7.core.Context  REQUIRED. For yes/no scoring.
---     judge_vocab        ion7.core.Vocab    REQUIRED.
---     extractor          { engine, cm }     REQUIRED. For claim extraction.
---     extract_max_tokens integer?  Default 512. Cap on extraction output.
---     judge_threshold    number?   Default 0. `logp(yes) - logp(no)`
---                  above this counts as supported.
--- }
--- @return Faithfulness
--- @raise When any required opt is missing.
function Faithfulness.new(opts)
    opts = opts or {}
    local Pointwise = require("ion7.rag.rerank.pointwise").Pointwise

    local judge = Pointwise.new({
        ctx          = assert(opts.judge_ctx,   "Faithfulness.new : opts.judge_ctx required"),
        vocab        = assert(opts.judge_vocab, "Faithfulness.new : opts.judge_vocab required"),
        system       = prompts.FAITHFULNESS_JUDGE_SYSTEM,
        user_format  = prompts.FAITHFULNESS_JUDGE_USER_FORMAT,
        max_doc_chars = 4000,
    })

    return setmetatable({
        _judge        = judge,
        _extractor    = assert(opts.extractor, "Faithfulness.new : opts.extractor required"),
        _extract_max  = opts.extract_max_tokens or 512,
        _threshold    = opts.judge_threshold     or 0,
    }, Faithfulness)
end

--- Compute the faithfulness score for an answer against its
--- supporting contexts.
---
--- @param  answer    string | Response       The model's answer.
--- @param  contexts  string | string[] | Hit[]  Excerpts the answer
---                  was grounded on. Strings, arrays of strings, and
---                  hit-shaped tables (`raw_text` / `text` / `content`)
---                  are all accepted.
--- @return number  Score in [0, 1]. Returns 1.0 (vacuous truth) when
---                  no atomic claims could be extracted.
--- @return table  Details : `{ claims : string[], scores : number[],
---                  n_supported : integer, note? : string }`.
function Faithfulness:score(answer, contexts)
    local answer_text = _coerce_answer(answer)
    local context_text = _join_contexts(contexts)

    -- Stage 1 : claim extraction via the extractor engine.
    local llm = require "ion7.llm"
    local engine = self._extractor.engine
    local cm     = self._extractor.cm
    cm:set_system(prompts.CLAIM_EXTRACT_SYSTEM)
    local session = llm.Session.new()
    session:add_user(string.format(prompts.CLAIM_EXTRACT_USER_FORMAT, answer_text))
    local response = engine:chat(session, { max_tokens = self._extract_max })
    cm:release(session)

    local claims = _parse_claims(response.content or "")
    if #claims == 0 then
        -- Without claims, scoring is undefined ; return 1.0 by vacuous
        -- truth and surface the absence of claims in `details.note`
        -- so callers can warn or re-run with a richer extractor.
        return 1.0, { claims = {}, scores = {}, n_supported = 0,
                      note = "no atomic claims extracted from answer" }
    end

    -- Stage 2 : yes/no score per claim.
    local scores = {}
    local n_supported = 0
    for i, claim in ipairs(claims) do
        scores[i] = self._judge:score(context_text, claim)
        if scores[i] > self._threshold then
            n_supported = n_supported + 1
        end
    end

    return n_supported / #claims, {
        claims      = claims,
        scores      = scores,
        n_supported = n_supported,
    }
end

return M
