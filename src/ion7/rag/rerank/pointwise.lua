--- @module ion7.rag.rerank.pointwise
--- @author  ion7 / Ion7 Project Contributors
---
--- Pointwise LLM-as-judge reranker. For each `(query, document)`
--- pair, the reranker renders a chat-template prompt that ends right
--- before the assistant's first token, decodes it through the
--- context, and reads
---
---   score(q, d) = logp(yes_token | prompt) - logp(no_token | prompt)
---
--- The score is a continuous log-odds — higher = more relevant.
---
--- Qwen3-Reranker (June 2025) was trained for this exact pattern, and
--- the same code path covers any chat-tuned LLM as a zero-shot
--- reranker at some quality cost. The prompt template and yes/no
--- token words are swappable via `opts`.
---
--- The reranker operates on a SINGLE-SEQUENCE context. Parallel
--- reranking across many queries needs several Pointwise instances on
--- independent contexts.

local chunks_db = require "ion7.rag.db.chunks"

local M = {}

-- ── Defaults ────────────────────────────────────────────────────────────

local DEFAULT_SYSTEM = [[Determine if the Document is relevant to the Query. Answer only "yes" or "no".]]

local DEFAULT_USER_FORMAT = [[Query: %s

Document: %s

Answer:]]

local DEFAULT_YES = " yes"
local DEFAULT_NO  = " no"

-- ── Helpers ─────────────────────────────────────────────────────────────

--- Tokenise `word` and return the FIRST token id. Most chat-model
--- tokenisers fold a leading-space single word like " yes" into one
--- token ; some split it. Taking token 0 keeps the score readable
--- across either tokenisation.
local function _first_token(vocab, word)
    local tokens, n = vocab:tokenize(word, false, false)
    if not tokens or n == 0 then
        error("ion7.rag.rerank.pointwise : tokeniser returned empty for '" ..
              tostring(word) .. "'", 3)
    end
    return tonumber(tokens[0])
end

-- ── Class ───────────────────────────────────────────────────────────────

local Pointwise = {}
Pointwise.__index = Pointwise
M.Pointwise = Pointwise

--- Construct a Pointwise reranker.
---
--- @param  opts table {
---     ctx           ion7.core.Context  REQUIRED. Single-sequence context.
---     vocab         ion7.core.Vocab    REQUIRED.
---     system        string?            Custom system prompt.
---     user_format   string?            "%s" placeholders for query, doc.
---     yes_word      string?            Default " yes". The leading space
---                                      matters — most BPE tokenisers
---                                      tokenise word boundaries
---                                      differently.
---     no_word       string?            Default " no".
---     max_doc_chars integer?           Truncate documents above this length
---                                      (default 2000) so a single rerank
---                                      call cannot overflow the context.
--- }
--- @return Pointwise
function Pointwise.new(opts)
    opts = opts or {}
    local ctx   = assert(opts.ctx,   "Pointwise.new : opts.ctx is required")
    local vocab = assert(opts.vocab, "Pointwise.new : opts.vocab is required")

    local self = setmetatable({
        _ctx           = ctx,
        _vocab         = vocab,
        _system        = opts.system        or DEFAULT_SYSTEM,
        _user_format   = opts.user_format   or DEFAULT_USER_FORMAT,
        _yes_id        = _first_token(vocab, opts.yes_word or DEFAULT_YES),
        _no_id         = _first_token(vocab, opts.no_word  or DEFAULT_NO),
        _max_doc_chars = opts.max_doc_chars or 2000,
    }, Pointwise)
    return self
end

--- Score a single (query, document) pair. Returns the log-odds of the
--- model emitting "yes" vs "no" as the first token of its answer.
--- Higher = more relevant.
---
--- @param  query    string
--- @param  document string
--- @return number
function Pointwise:score(query, document)
    if #document > self._max_doc_chars then
        document = document:sub(1, self._max_doc_chars)
    end

    local messages = {
        { role = "system", content = self._system },
        { role = "user",   content = string.format(self._user_format,
                                                   query, document) },
    }

    -- Chat template rendered with a generation prompt so the last
    -- token of the rendered prompt sits right before the assistant's
    -- first output token — the exact position to read logits from.
    local prompt = self._vocab:apply_template(messages, true, -1)
    local tokens, n = self._vocab:tokenize(prompt, false, true)

    -- Fresh KV row per scoring call. Reranking workloads in this
    -- module are latency-bound on the critical path, so prompt cost
    -- amortises poorly across documents and the simpler clear-then-
    -- decode is preferred.
    self._ctx:kv_clear()
    self._ctx:decode(tokens, n)

    local last_idx = n - 1
    local logp_yes = self._ctx:logprob(last_idx, self._yes_id)
    local logp_no  = self._ctx:logprob(last_idx, self._no_id)

    return logp_yes - logp_no
end

--- Rerank an existing Hit[] (typically the output of
--- `ion7.rag.retrieve.search`) by scoring each chunk's `raw_text`
--- against `query`.
---
--- The handle is used to hydrate `raw_text` from `chunks.db`. The
--- returned hits carry the original score under `prior_score` and the
--- new reranker score under `score`.
---
--- @param  handle  ion7.rag.db.Handle
--- @param  query   string
--- @param  hits    Hit[]      Each entry must carry `chunk_id`.
--- @param  k       integer?   Top-k to keep, default `#hits`.
--- @return Hit[]   `{ { chunk_id, score, prior_score, raw_text }, ... }`
---                  sorted by descending `score`.
function Pointwise:rerank(handle, query, hits, k)
    local out = {}
    for i, hit in ipairs(hits) do
        local row = chunks_db.get_chunk(handle, hit.chunk_id)
        if row then
            out[#out + 1] = {
                chunk_id    = hit.chunk_id,
                prior_score = hit.score,
                score       = self:score(query, row.raw_text),
                raw_text    = row.raw_text,
            }
        end
    end

    table.sort(out, function(a, b) return a.score > b.score end)

    if k and k < #out then
        local trimmed = {}
        for i = 1, k do trimmed[i] = out[i] end
        return trimmed
    end
    return out
end

return M
