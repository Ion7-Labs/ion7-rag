--- @module ion7.rag.context
--- @author  ion7 / Ion7 Project Contributors
---
--- Anthropic-style Contextual Retrieval (Sept 2024). For each chunk,
--- a small fast LLM is prompted with the whole document plus the
--- chunk excerpt and asked for ~50-100 tokens that situate the
--- excerpt within the document. The generated context is prepended to
--- the chunk before embedding and FTS indexing.
---
--- Reported gains : ~35 % retrieval-failure reduction (Anthropic
--- blog), ~49 % with a reranker on top. Pham et al., arXiv:2504.19754
--- (ECIR 2025) confirms the headline number and the 4:1 dense:BM25
--- fusion ratio.
---
--- The Enricher class drives `ion7-llm.Engine` once per chunk :
---   - Each enrichment is an independent chat call with a fresh
---     `Session`, so chunks never bleed history into one another.
---   - The Engine renders the chat template and runs the generation
---     loop ; the Enricher only collects `response.content`.
---
--- Caller owns the Context and Vocab handles. The Enricher attaches
--- its own `kv.ContextManager` and `Engine` on top, sets the system
--- prompt once, and reuses them across enrichment calls.

local prompts = require "ion7.rag.context.prompts"

local M = {}
M.prompts = prompts

-- ── Class ───────────────────────────────────────────────────────────────

local Enricher = {}
Enricher.__index = Enricher
M.Enricher = Enricher

--- Construct an Enricher.
---
--- @param  opts table {
---     ctx          ion7.core.Context     REQUIRED. Single-sequence context.
---     vocab        ion7.core.Vocab       REQUIRED.
---     system       string?               Override the system prompt.
---     user_format  string?               Override the user template ;
---                                        must carry two `%s` placeholders
---                                        for `(full_doc, chunk)`.
---     max_tokens   integer?              Cap on generated tokens. Default
---                                        128 (~100 tokens of useful
---                                        context plus small slack).
---     max_doc_chars integer?             Truncate the full document above
---                                        this many characters before
---                                        sending. Default 20000 (~5K
---                                        tokens) ; the caller is
---                                        responsible for not feeding in a
---                                        novel.
---     sampler      ion7.core.Sampler?    Custom sampler ; default is the
---                                        ion7-llm "precise" profile (low
---                                        temperature, top_k 20, factual
---                                        / extractive).
---     headroom     integer?              Engine KV headroom. Default 256.
--- }
--- @return Enricher
--- @raise When `ctx` or `vocab` is missing.
function Enricher.new(opts)
    opts = opts or {}
    local ctx   = assert(opts.ctx,   "context.Enricher.new : opts.ctx required")
    local vocab = assert(opts.vocab, "context.Enricher.new : opts.vocab required")

    local llm = require "ion7.llm"

    local cm, engine = llm.pipeline(ctx, vocab, {
        headroom = opts.headroom or 256,
    })
    cm:set_system(opts.system or prompts.SYSTEM)

    local sampler = opts.sampler
    if not sampler then
        local profiles = require "ion7.llm.sampler.profiles"
        sampler = profiles.precise()
    end

    return setmetatable({
        _ctx           = ctx,
        _vocab         = vocab,
        _llm           = llm,
        _cm            = cm,
        _engine        = engine,
        _sampler       = sampler,
        _user_format   = opts.user_format   or prompts.USER_FORMAT,
        _max_tokens    = opts.max_tokens    or 128,
        _max_doc_chars = opts.max_doc_chars or 20000,
    }, Enricher)
end

--- Generate the contextual prefix for a single `(full_doc, chunk)`
--- pair. The model output is returned verbatim with leading / trailing
--- whitespace trimmed.
---
--- @param  full_doc_text  string
--- @param  chunk_text     string
--- @return string  Generated context, trimmed. Empty string when the
---                  model produces nothing.
function Enricher:enrich_chunk(full_doc_text, chunk_text)
    if #full_doc_text > self._max_doc_chars then
        full_doc_text = full_doc_text:sub(1, self._max_doc_chars)
    end

    local user = string.format(self._user_format, full_doc_text, chunk_text)

    local session = self._llm.Session.new()
    session:add_user(user)

    local response = self._engine:chat(session, {
        max_tokens = self._max_tokens,
        sampler    = self._sampler,
    })

    -- Release the seq slot before returning. Each enrichment is
    -- independent and the session is not retained for fast-path
    -- snapshots, so the slot is recycled. Without this, `n_seq_max`
    -- would have to grow with the number of chunks ever processed.
    self._cm:release(session)

    local out = response.content or ""
    out = out:gsub("^%s+", ""):gsub("%s+$", "")
    return out
end

--- Enrich an array of chunks against a single document. Mutates each
--- chunk in place by setting `chunk.contextual_text` to
--- `<context>\n\n<raw_text>` (blank-line-joined) so downstream
--- embedders and FTS see the prepended view by default. Returns the
--- same array for chaining.
---
--- @param  full_doc_text  string
--- @param  chunks         Chunk[]
--- @param  on_progress    function(done, total)?  Called after every
---                  enrichment with 1-based progress.
--- @return Chunk[]  The input array, mutated in place.
function Enricher:enrich_chunks(full_doc_text, chunks, on_progress)
    for i, chunk in ipairs(chunks) do
        local ctx = self:enrich_chunk(full_doc_text, chunk.raw_text)
        if ctx == "" then
            chunk.contextual_text = chunk.raw_text
        else
            chunk.contextual_text = ctx .. "\n\n" .. chunk.raw_text
        end
        if on_progress then on_progress(i, #chunks) end
    end
    return chunks
end

--- Drop the per-Enricher engine and ContextManager refs. The
--- underlying Context and Vocab are NOT freed ; the caller retains
--- ownership. Engine / ContextManager hold no native resources of
--- their own beyond what the Context already manages, so dropping
--- the refs lets a caller-side `:free` on the Context finalise
--- cleanly.
function Enricher:close()
    self._engine = nil
    self._cm     = nil
end

return M
