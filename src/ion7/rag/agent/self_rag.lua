--- @module ion7.rag.agent.self_rag
--- @author  ion7 / Ion7 Project Contributors
---
--- Self-RAG (Asai et al., arXiv:2310.11511, 2023).
---
--- The original paper trains a model that emits special reflection
--- tokens at decision points : `[Retrieve]` / `[No Retrieve]`,
--- `[Relevant]` / `[Irrelevant]`, `[Supported]` / `[Partially
--- Supported]`. The same control flow runs on any generic chat model
--- by constraining each decision to a strict JSON schema via
--- `ion7.grammar` — the grammar masks invalid tokens at sample time,
--- so malformed reflection output is impossible by construction.
---
--- Three reflection points :
---
---   1. retrieve_decision  `{retrieve: bool, rationale: string}`
---   2. relevance_grade    `{relevant: bool, rationale: string}`  (per hit)
---   3. support_grade      `{supported: "full"|"partial"|"none",
---                            rationale: string}`  (post-answer)

local prompts = require "ion7.rag.agent.prompts"

local M = {}

-- ── Defaults ────────────────────────────────────────────────────────────

M.DEFAULTS = {
    grade_top_k        = 5,
    min_relevant_hits  = 1,   -- if 0 hits pass relevance, fall back to all
    drop_irrelevant    = true,
}

-- ── JSON-schema-driven sampler builders ─────────────────────────────────

local function _build_grammar(schema_table)
    local Grammar = require "ion7.grammar"
    local g = Grammar.from_type(schema_table)
    return g:to_gbnf()
end

local function _build_sampler(ion7, vocab, gbnf, opts)
    opts = opts or {}
    -- Sampler ordering matters with constrained generation : grammar
    -- masks invalid tokens against the FULL vocab and must come BEFORE
    -- any candidate-pruning step (top_p, top_k, temp). Placing grammar
    -- after a pruner like top_p risks the pruner shrinking the candidate
    -- pool to a set the grammar then rejects entirely, which surfaces
    -- as a C++ exception out of the sampler chain. The canonical
    -- DCCD-style order is : grammar → top_k → dist.
    return ion7.Sampler.chain()
        :grammar(gbnf, "root", vocab)
        :top_k(opts.top_k or 40)
        :temp(opts.temp or 0.3)
        :dist()
        :build()
end

-- ── JSON helper (uses the json bundled by ion7-core) ────────────────────

local function _decode_json(text)
    local json = require "ion7.vendor.json"
    -- The sampler guarantees valid-shape JSON, but the model can
    -- still wrap it in whitespace, code fences, or a trailing
    -- newline. Strip those before decoding.
    local clean = text:gsub("```json", ""):gsub("```", "")
                      :gsub("^%s+", ""):gsub("%s+$", "")
    local ok, val = pcall(json.decode, clean)
    if ok then return val end
    return nil
end

-- ── Class ───────────────────────────────────────────────────────────────

local SelfRAG = {}
SelfRAG.__index = SelfRAG
M.SelfRAG = SelfRAG

--- Construct a Self-RAG agent.
---
--- @param  opts table {
---     pipeline          ion7.rag.Pipeline   REQUIRED.
---     judge             { engine, cm }?    Engine pair for
---                  grammar-constrained reflection. Defaults to
---                  `pipeline.answerer`.
---     drop_irrelevant   bool?       Default true. Drop hits the model
---                  grades as irrelevant before answering.
---     min_relevant_hits integer?    Default 1. When fewer than this
---                  many hits pass the relevance check, fall back to
---                  all hits to avoid empty-context generation.
---     grade_top_k       integer?    Default 5. Number of hits to grade
---                  for relevance and to summarise for support grading.
--- }
--- @return SelfRAG
--- @raise When `pipeline`, the resolved judge pair, or its `vocab` is
---        missing.
function SelfRAG.new(opts)
    opts = opts or {}
    local pipe = assert(opts.pipeline, "agent.SelfRAG.new : opts.pipeline required")

    local judge = opts.judge or pipe._answerer
    if not judge then
        error("agent.SelfRAG.new : a judge engine pair is required " ..
              "(opts.judge or pipeline.answerer)", 2)
    end

    -- Resolve the vocab the judge engine speaks via the engine's
    -- `_vocab` (or `vocab`) field. Pipeline answerers wired by
    -- `Pipeline.new` carry one of these references.
    local vocab = (judge.engine and judge.engine._vocab)
                or (judge.engine and judge.engine.vocab)
    if not vocab then
        error("agent.SelfRAG.new : cannot resolve vocab from judge.engine ; " ..
              "pass opts.vocab explicitly", 2)
    end

    local ion7 = require "ion7.core"

    -- Pre-build the three GBNF strings + samplers.
    local gbnf_retrieve = _build_grammar({
        retrieve  = "boolean",
        rationale = "string",
    })
    local gbnf_relevant = _build_grammar({
        relevant  = "boolean",
        rationale = "string",
    })
    local gbnf_supported = _build_grammar({
        supported = { enum = { "full", "partial", "none" } },
        rationale = "string",
    })

    return setmetatable({
        _pipe              = pipe,
        _judge             = judge,
        _vocab             = vocab,
        _ion7              = ion7,
        _sampler_retrieve  = _build_sampler(ion7, vocab, gbnf_retrieve),
        _sampler_relevant  = _build_sampler(ion7, vocab, gbnf_relevant),
        _sampler_supported = _build_sampler(ion7, vocab, gbnf_supported),
        _grade_top_k       = opts.grade_top_k       or M.DEFAULTS.grade_top_k,
        _min_relevant_hits = opts.min_relevant_hits or M.DEFAULTS.min_relevant_hits,
        _drop_irrelevant   = (opts.drop_irrelevant ~= false),
    }, SelfRAG)
end

-- ── Internal : run a single grammar-constrained reflection ──────────────

function SelfRAG:_reflect(system, user, sampler)
    local llm = require "ion7.llm"
    local engine = self._judge.engine
    local cm     = self._judge.cm
    cm:set_system(system)
    local session = llm.Session.new()
    session:add_user(user)
    local response = engine:chat(session, {
        max_tokens = 192,
        sampler    = sampler,
    })
    cm:release(session)
    return _decode_json(response.content or "")
end

-- ── Public ──────────────────────────────────────────────────────────────

--- Run the Self-RAG loop for `query`.
---
--- @param  query  string
--- @param  opts   table? {
---     retrieve  table?  Pass-through to `Pipeline:retrieve`.
---     ask       table?  Pass-through to the answerer when retrieval
---                  branch fires.
---     direct_answer_max_tokens integer?  Default 256. Token cap when
---                  the model votes no-retrieval and the answerer is
---                  called directly.
--- }
--- @return Response  Final answer.
--- @return table  Reflection log :
---                  `{ decisions = { retrieve, relevance[], support } }`.
function SelfRAG:run(query, opts)
    opts = opts or {}
    local log = {
        decisions = {},
    }

    -- Step 1 : decide whether to retrieve. The grammar makes
    -- valid-shape JSON the only sampleable output, but a stop-string
    -- or zero-length response can still slip through on small models
    -- at low but non-zero rate. The fallback on parse failure is
    -- `retrieve = true` — an unnecessary retrieve is cheaper than
    -- answering blind.
    local d_retrieve = self:_reflect(
        prompts.SELFRAG_RETRIEVE_SYSTEM,
        string.format(prompts.SELFRAG_RETRIEVE_USER_FORMAT, query),
        self._sampler_retrieve)
    if d_retrieve == nil then
        d_retrieve = {
            retrieve  = true,
            rationale = "(reflection parse failed ; defaulting to retrieve)",
        }
    end
    log.decisions.retrieve = d_retrieve

    if d_retrieve.retrieve == false then
        -- No-retrieval branch : ask the model directly.
        local llm = require "ion7.llm"
        local engine = self._pipe._answerer and self._pipe._answerer.engine or self._judge.engine
        local cm     = self._pipe._answerer and self._pipe._answerer.cm     or self._judge.cm
        cm:set_system("Answer the user's question concisely.")
        local session = llm.Session.new()
        session:add_user(query)
        local response = engine:chat(session, {
            max_tokens = opts.direct_answer_max_tokens or 256,
        })
        cm:release(session)
        return response, log
    end

    -- Step 2 : retrieve and grade.
    local hits = self._pipe:retrieve(query, opts.retrieve)
    log.decisions.relevance = {}

    local kept = {}
    local k = math.min(#hits, self._grade_top_k)
    for i = 1, k do
        local d = self:_reflect(
            prompts.SELFRAG_RELEVANT_SYSTEM,
            string.format(prompts.SELFRAG_RELEVANT_USER_FORMAT,
                          query, hits[i].raw_text),
            self._sampler_relevant)
        log.decisions.relevance[i] = d
        if not self._drop_irrelevant or (d and d.relevant) then
            kept[#kept + 1] = hits[i]
        end
    end
    if #kept < self._min_relevant_hits then
        -- Fall back to all hits to avoid empty-context generation.
        kept = hits
    end

    -- Step 3 : generate the answer with the surviving (relevant) hits.
    local response = self:_answer_with_hits(query, kept, opts.ask)

    -- Step 4 : grade support.
    local excerpts = {}
    local n_excerpts = math.min(#kept, self._grade_top_k)
    for i = 1, n_excerpts do
        excerpts[i] = string.format("[%d] %s", i, kept[i].raw_text)
    end
    local d_support = self:_reflect(
        prompts.SELFRAG_SUPPORTED_SYSTEM,
        string.format(prompts.SELFRAG_SUPPORTED_USER_FORMAT,
            query, table.concat(excerpts, "\n\n"), response.content or ""),
        self._sampler_supported)
    log.decisions.support = d_support

    return response, log
end

--- Generate from a pre-filtered hit list. Same shape as
--- `CRAG:_answer_with_hits` minus the confidence caveats — Self-RAG
--- has already filtered for relevance.
function SelfRAG:_answer_with_hits(query, hits, ask_opts)
    ask_opts = ask_opts or {}
    local pipe = self._pipe
    if not pipe._answerer then
        error("agent.SelfRAG : Pipeline has no answerer ; cannot generate", 2)
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
