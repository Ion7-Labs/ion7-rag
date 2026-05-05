--- @module ion7.rag.agent.crag
--- @author  ion7 / Ion7 Project Contributors
---
--- Corrective Retrieval-Augmented Generation (Yan et al., 2024).
--- Wraps a `ion7.rag.Pipeline` with a relevance-driven control loop :
---
---   1. Retrieve candidates for the original query.
---   2. Score the top-K candidates with the configured reranker.
---   3. Pick a confidence tier from the scores :
---        high  (max ≥ threshold_correct)    use as-is
---        low   (max ≤ threshold_incorrect)  reformulate the query and
---                                            retrieve again, up to
---                                            `max_retries` times
---        mixed                              use what came back
---   4. Hand the surviving hits to the Pipeline answerer with a
---      caveat instruction when confidence is mixed, or after retries
---      are exhausted on a low-confidence branch.
---
--- The agent never mutates the underlying Pipeline ; it only
--- orchestrates calls. CRAG-specific opts (thresholds, max_retries,
--- reformulator) live on the agent. Pipeline opts pass through
--- opaquely.

local prompts = require "ion7.rag.agent.prompts"

local M = {}

-- ── Defaults ────────────────────────────────────────────────────────────

M.DEFAULTS = {
    threshold_correct   = 0.5,
    threshold_incorrect = -0.5,
    max_retries         = 1,
    eval_top_k          = 5,
}

-- ── Class ───────────────────────────────────────────────────────────────

local CRAG = {}
CRAG.__index = CRAG
M.CRAG = CRAG

--- Construct a CRAG agent.
---
--- @param  opts table {
---     pipeline                ion7.rag.Pipeline   REQUIRED.
---     reranker                {:score(q, doc)}?   Defaults to
---                                       `pipeline.reranker`. Required
---                                       either via opts or at Pipeline
---                                       construction.
---     reformulator            { engine, cm }?     Engine pair used to
---                                       rephrase the query on
---                                       low-confidence branches.
---                                       Defaults to `pipeline.answerer`.
---     reformulate_system      string?  Override `prompts.REFORMULATE_SYSTEM`.
---     reformulate_user_format string?  Override `prompts.REFORMULATE_USER_FORMAT`.
---     threshold_correct       number?  Default 0.5.
---     threshold_incorrect     number?  Default -0.5.
---     max_retries             integer? Default 1.
---     eval_top_k              integer? Default 5. Number of top hits
---                                       fed into relevance evaluation.
--- }
--- @return CRAG
--- @raise When `pipeline`, the resolved reranker, or the resolved
---        reformulator is missing.
function CRAG.new(opts)
    opts = opts or {}
    local pipe = assert(opts.pipeline,
        "agent.CRAG.new : opts.pipeline required")

    local reranker = opts.reranker or pipe._reranker
    if not reranker then
        error("agent.CRAG.new : a reranker is required (pass opts.reranker " ..
              "or configure the Pipeline with one)", 2)
    end

    local reformulator = opts.reformulator or pipe._answerer
    if not reformulator then
        error("agent.CRAG.new : a reformulator engine is required " ..
              "(opts.reformulator or pipeline.answerer)", 2)
    end

    return setmetatable({
        _pipe                  = pipe,
        _reranker              = reranker,
        _reformulator          = reformulator,
        _reformulate_system    = opts.reformulate_system    or prompts.REFORMULATE_SYSTEM,
        _reformulate_user_fmt  = opts.reformulate_user_format or prompts.REFORMULATE_USER_FORMAT,
        _threshold_correct     = opts.threshold_correct   or M.DEFAULTS.threshold_correct,
        _threshold_incorrect   = opts.threshold_incorrect or M.DEFAULTS.threshold_incorrect,
        _max_retries           = opts.max_retries         or M.DEFAULTS.max_retries,
        _eval_top_k            = opts.eval_top_k          or M.DEFAULTS.eval_top_k,
    }, CRAG)
end

-- ── Internal helpers ────────────────────────────────────────────────────

--- Score the top-K hits and return (max_score, n_correct,
--- n_incorrect). Mutates `hits[i].score` in place to the reranker's
--- score so downstream consumers see the corrective evaluation.
function CRAG:_evaluate(query, hits)
    local k = math.min(#hits, self._eval_top_k)
    local max_score = -math.huge
    local n_correct, n_incorrect = 0, 0
    for i = 1, k do
        local s = self._reranker:score(query, hits[i].raw_text)
        hits[i].score = s
        if s > max_score then max_score = s end
        if s >= self._threshold_correct       then n_correct   = n_correct   + 1
        elseif s <= self._threshold_incorrect then n_incorrect = n_incorrect + 1 end
    end
    return max_score, n_correct, n_incorrect
end

--- Reformulate the query via the LLM. Returns the rephrased string.
function CRAG:_reformulate(query)
    local llm = require "ion7.llm"
    local engine = self._reformulator.engine
    local cm     = self._reformulator.cm
    cm:set_system(self._reformulate_system)

    local session = llm.Session.new()
    session:add_user(string.format(self._reformulate_user_fmt, query))
    local response = engine:chat(session, { max_tokens = 96 })
    cm:release(session)

    local out = response.content or ""
    out = out:gsub("^%s+", ""):gsub("%s+$", "")
    -- Strip a matched outer quote pair when chat models wrap the
    -- rephrased question.
    out = out:match("^['\"](.-)['\"]$") or out
    return out
end

-- ── Public ──────────────────────────────────────────────────────────────

--- Run the CRAG control loop for `query`.
---
--- @param  query  string
--- @param  opts   table? {
---     retrieve  table?  Pass-through to `Pipeline:retrieve`.
---     ask       table?  Pass-through to the answerer (top_k, sampler,
---                  max_tokens).
--- }
--- @return Response  The final answer.
--- @return table  Metadata : `{ retrievals : integer, max_scores :
---                  number[], queries : string[], confidence : "high"
---                  | "mixed" | "low" }`.
function CRAG:run(query, opts)
    opts = opts or {}

    local meta = {
        retrievals  = 0,
        max_scores  = {},
        queries     = { query },
        confidence  = nil,
    }

    local current_query = query
    local hits          = self._pipe:retrieve(current_query, opts.retrieve)
    meta.retrievals = meta.retrievals + 1
    local max_s, n_correct, n_incorrect = self:_evaluate(current_query, hits)
    meta.max_scores[#meta.max_scores + 1] = max_s

    -- Retry loop while confidence is low and budget remains.
    local retries = 0
    while max_s <= self._threshold_incorrect and retries < self._max_retries do
        retries = retries + 1
        current_query = self:_reformulate(query)
        meta.queries[#meta.queries + 1] = current_query

        hits = self._pipe:retrieve(current_query, opts.retrieve)
        meta.retrievals = meta.retrievals + 1
        max_s, n_correct, n_incorrect = self:_evaluate(current_query, hits)
        meta.max_scores[#meta.max_scores + 1] = max_s
    end

    -- Tag final confidence band.
    if max_s >= self._threshold_correct then
        meta.confidence = "high"
    elseif max_s <= self._threshold_incorrect then
        meta.confidence = "low"
    else
        meta.confidence = "mixed"
    end

    -- Re-rank hits by the corrective scores (already mutated above).
    table.sort(hits, function(a, b)
        return (a.score or -math.huge) > (b.score or -math.huge)
    end)

    -- Generation reuses the answerer the Pipeline already owns ; CRAG
    -- does not run its own augmentation prompt and trusts the
    -- Pipeline's template. The corrected hits are injected by
    -- bypassing `Pipeline:ask`'s internal `:retrieve` call and
    -- replicating the augmentation/answer tail inline.
    local response = self:_answer_with_hits(current_query, hits, opts.ask, meta.confidence)
    return response, meta
end

--- Generate from a pre-computed hit list. Mirrors the tail of
--- `Pipeline:ask` and applies the CRAG confidence caveat when needed.
function CRAG:_answer_with_hits(query, hits, ask_opts, confidence)
    ask_opts = ask_opts or {}
    local pipe = self._pipe
    if not pipe._answerer then
        error("agent.CRAG : Pipeline has no answerer ; cannot generate", 2)
    end

    local llm = require "ion7.llm"
    local engine = pipe._answerer.engine
    local cm     = pipe._answerer.cm

    local top_k = ask_opts.top_k or pipe._augment_top_k
    local n = math.min(#hits, top_k)
    local parts = {}
    for i = 1, n do
        local h = hits[i]
        local cite = h.doc_id or ("chunk-" .. tostring(h.chunk_id))
        if h.section and h.section ~= "" then cite = cite .. " > " .. h.section end
        parts[i] = string.format("[%d] (%s)\n%s", i, cite, h.raw_text)
    end

    local prompt = string.format(pipe._augment_template,
        table.concat(parts, "\n\n"), query)
    if confidence == "low" then
        prompt = prompt .. "\n\nNote : retrieval confidence was low ; if no excerpt directly answers the question, say so explicitly."
    elseif confidence == "mixed" then
        prompt = prompt .. "\n\nNote : retrieval confidence was mixed ; only state facts that an excerpt clearly supports."
    end

    cm:set_system(pipe._answer_system)
    local session = llm.Session.new()
    session:add_user(prompt)
    local response = engine:chat(session, {
        max_tokens = ask_opts.max_tokens or 512,
        sampler    = ask_opts.sampler,
    })
    cm:release(session)
    return response
end

return M
