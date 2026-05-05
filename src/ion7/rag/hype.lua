--- @module ion7.rag.hype
--- @author  ion7 / Ion7 Project Contributors
---
--- HyPE — Hypothetical Prompt Embeddings.
---
--- For each chunk, prompt a small fast LLM to produce N hypothetical
--- questions that the chunk would be a good answer to. Those
--- questions are then embedded and stored in `idx.hype_vec` with a
--- back-reference to the parent chunk. At query time,
--- `ion7.rag.retrieve.search` searches both the chunk vectors and the
--- HyPE-question vectors, and collapses the per-chunk best into a
--- single dense candidate list (see
--- `ion7.rag.db.hype.collapse_to_chunks`).
---
--- Effect : question-shaped queries match question-shaped vectors
--- much more cleanly than they match doc-style chunk vectors, which
--- typically lifts P@k by tens of points on factoid-heavy workloads.
---
--- Lifecycle : the caller supplies an `ion7.core.Context` and a
--- matching `ion7.core.Vocab` ; the Generator instantiates its own
--- `kv.ContextManager` + `Engine` over those, and releases sessions
--- after every `:generate` call so a single `n_seq_max = 1` context
--- handles unbounded ingest volume.

local M = {}

-- ── Default prompts ─────────────────────────────────────────────────────

M.SYSTEM = [[You are a question-design assistant. Given a passage, list N short, distinct, natural-language questions for which the passage would be a good answer. Use the same language as the passage. Do not paraphrase the passage ; produce questions a real user would type. Reply with a numbered list, one question per line, nothing else.]]

M.USER_FORMAT = [[Passage :
%s

Produce exactly %d questions, numbered 1 to %d.]]

-- ── Helpers ─────────────────────────────────────────────────────────────

--- Parse a free-form model response into an array of question strings.
--- Recognised line prefixes : `1.` / `1)` (numbered), `-` / `*`
--- (bulleted), `Q:` / `Q1-` (interrogative). Whitespace, leading
--- markdown emphasis (`*` `` ` ``) and exact duplicates are stripped.
--- Lines shorter than 4 characters are dropped as malformed.
local function _parse_questions(text)
    local seen = {}
    local out  = {}
    for line in (text or ""):gmatch("[^\n]+") do
        local q = line:match("^%s*%d+[%.%)]%s*(.+)$")
              or line:match("^%s*[%-%*]%s+(.+)$")
              or line:match("^%s*Q%d*%s*[%:%-]%s*(.+)$")
        if q then
            q = q:gsub("^[%s%*%`]+", ""):gsub("[%s%*%`]+$", "")
            if #q > 3 and not seen[q] then
                seen[q] = true
                out[#out + 1] = q
            end
        end
    end
    return out
end

-- ── Class ───────────────────────────────────────────────────────────────

local Generator = {}
Generator.__index = Generator
M.Generator = Generator

--- Build a HyPE Generator over a single chat-tuned context.
---
--- @param  opts table {
---     ctx              ion7.core.Context    REQUIRED. Single-seq is enough.
---     vocab            ion7.core.Vocab      REQUIRED.
---     n_questions      integer?             Default number of questions
---                       per chunk. Overridable per `:generate` call.
---                       Default 3.
---     system           string?              Override the system prompt.
---     user_format      string?              Override the user template.
---                       Must carry three `%s` placeholders for
---                       `(passage, n_questions, n_questions)`.
---     max_tokens       integer?             Generation cap. Default 192.
---     max_chunk_chars  integer?             Truncate any chunk longer
---                       than this many characters before prompting.
---                       Default 4000.
---     sampler          ion7.core.Sampler?   Override the default sampler.
---                       Defaults to `ion7.llm.sampler.profiles.balanced()`,
---                       which keeps a small temperature ; this nudges the
---                       model toward distinct questions rather than
---                       rephrasings of a single one.
---     headroom         integer?             KV headroom passed to
---                       `llm.pipeline`. Default 256.
--- }
--- @return ion7.rag.hype.Generator
function Generator.new(opts)
    opts = opts or {}
    local ctx   = assert(opts.ctx,   "hype.Generator.new : opts.ctx required")
    local vocab = assert(opts.vocab, "hype.Generator.new : opts.vocab required")

    local llm = require "ion7.llm"

    local cm, engine = llm.pipeline(ctx, vocab, {
        headroom = opts.headroom or 256,
    })
    cm:set_system(opts.system or M.SYSTEM)

    local sampler = opts.sampler
    if not sampler then
        local profiles = require "ion7.llm.sampler.profiles"
        sampler = profiles.balanced()
    end

    return setmetatable({
        _ctx             = ctx,
        _vocab           = vocab,
        _llm             = llm,
        _cm              = cm,
        _engine          = engine,
        _sampler         = sampler,
        _user_format     = opts.user_format     or M.USER_FORMAT,
        _n_questions     = opts.n_questions     or 3,
        _max_tokens      = opts.max_tokens      or 192,
        _max_chunk_chars = opts.max_chunk_chars or 4000,
    }, Generator)
end

--- Generate hypothetical questions for one chunk.
---
--- The returned array can be shorter than `n_questions` if the model's
--- output cannot be parsed into that many distinct lines — questions
--- are never invented to fill the quota, never duplicated, and a
--- partial response never raises.
---
--- @param  chunk_text   string
--- @param  n_questions  integer?  Override the constructor default.
--- @return string[]                Zero or more parsed questions.
function Generator:generate(chunk_text, n_questions)
    n_questions = n_questions or self._n_questions
    if #chunk_text > self._max_chunk_chars then
        chunk_text = chunk_text:sub(1, self._max_chunk_chars)
    end

    local user = string.format(self._user_format,
        chunk_text, n_questions, n_questions)

    local session = self._llm.Session.new()
    session:add_user(user)
    local response = self._engine:chat(session, {
        max_tokens = self._max_tokens,
        sampler    = self._sampler,
    })
    self._cm:release(session)

    return _parse_questions(response.content)
end

--- Drop references to the engine and the context manager. The
--- underlying `ctx` and `vocab` remain owned by the caller and are
--- not freed.
function Generator:close()
    self._engine = nil
    self._cm     = nil
end

-- Exposed for tests and for external callers that want to parse a
-- known-shaped response without spinning up a Generator.
M._parse_questions = _parse_questions

return M
