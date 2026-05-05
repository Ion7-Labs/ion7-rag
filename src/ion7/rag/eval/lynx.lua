--- @module ion7.rag.eval.lynx
--- @author  ion7 / Ion7 Project Contributors
---
--- Lynx hallucination judge (Patronus AI, arXiv:2407.08488).
---
--- Lynx-8B / Lynx-70B are Llama-3 finetunes that score RAG outputs
--- against a `(question, document, answer)` triplet, returning PASS
--- or FAIL. Lynx is trained against a specific prompt template, which
--- this module mirrors verbatim. The same code path runs as a
--- bare-instruction zero-shot judge on any generic chat model ; the
--- intended pairing is a real Lynx GGUF (e.g.
--- PatronusAI/Patronus-Lynx-8B via gguf-my-repo).
---
--- Output : `"PASS"`, `"FAIL"`, or `"UNKNOWN"` when the margin sits
--- in the indecisive band.

local prompts = require "ion7.rag.eval.prompts"

local M = {}

local function _coerce_answer(a)
    if type(a) == "table" then return a.content or "" end
    return tostring(a or "")
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

-- ── Class ───────────────────────────────────────────────────────────────

local Lynx = {}
Lynx.__index = Lynx
M.Lynx = Lynx

--- Construct a Lynx judge.
---
--- @param  opts table {
---     ctx            ion7.core.Context  REQUIRED.
---     vocab          ion7.core.Vocab    REQUIRED.
---     system         string?  Override `prompts.LYNX_SYSTEM`.
---     user_format    string?  Override `prompts.LYNX_USER_FORMAT`.
---     yes_word       string?  Default `" PASS"`.
---     no_word        string?  Default `" FAIL"`.
---     max_doc_chars  integer? Default 6000. Per-document truncation
---                  before the (question, document, answer) prompt is
---                  assembled.
--- }
--- @return Lynx
function Lynx.new(opts)
    opts = opts or {}
    local Pointwise = require("ion7.rag.rerank.pointwise").Pointwise

    -- Lynx is a 3-input judge while Pointwise is a 2-input judge.
    -- The third input (the question) is baked into the user prompt
    -- when `:judge()` is called, leaving Pointwise to do the yes/no
    -- logprob read on the assembled prompt.
    return setmetatable({
        _ctx           = opts.ctx,
        _vocab         = opts.vocab,
        _system        = opts.system or prompts.LYNX_SYSTEM,
        _user_format   = opts.user_format or prompts.LYNX_USER_FORMAT,
        _yes_word      = opts.yes_word or " PASS",
        _no_word       = opts.no_word  or " FAIL",
        _Pointwise     = Pointwise,
        _max_doc_chars = opts.max_doc_chars or 6000,
    }, Lynx)
end

--- Judge whether `answer` is faithful to `contexts` given `question`.
---
--- @param  question  string
--- @param  answer    string | Response
--- @param  contexts  string | string[] | Hit[]  Excerpts the answer
---                  was grounded on.
--- @return string  Verdict : `"PASS"`, `"FAIL"`, or `"UNKNOWN"` when
---                  `|margin| ≤ 0.5`.
--- @return number  Margin = `logp(PASS) - logp(FAIL)`. Positive
---                  margins favour PASS.
function Lynx:judge(question, answer, contexts)
    local document = _join_contexts(contexts)
    if #document > self._max_doc_chars then
        document = document:sub(1, self._max_doc_chars)
    end
    local answer_text = _coerce_answer(answer)
    local prompt_user = string.format(self._user_format,
        question, document, answer_text)

    -- Reuse Pointwise for the yes/no logprob trick. The assembled
    -- 3-input prompt is passed through the first `%s` of a `%s%s`
    -- user_format ; the second placeholder consumes an empty string,
    -- leaving the assembled text intact. `max_doc_chars` is set
    -- above the prompt length so Pointwise never truncates it.
    local p = self._Pointwise.new({
        ctx           = self._ctx,
        vocab         = self._vocab,
        system        = self._system,
        user_format   = "%s%s",
        yes_word      = self._yes_word,
        no_word       = self._no_word,
        max_doc_chars = #prompt_user + 1,
    })
    local margin = p:score(prompt_user, "")

    local verdict = "UNKNOWN"
    if margin > 0.5 then verdict = "PASS"
    elseif margin < -0.5 then verdict = "FAIL"
    end
    return verdict, margin
end

return M
