--- @module ion7.rag.chunk.recursive
--- @author  ion7 / Ion7 Project Contributors
---
--- Recursive character splitter (FloTorch benchmark, NAACL 2025
--- Findings). Tries a hierarchy of separators from coarse (`\n\n`) to
--- fine (` `, ``), recursively splitting until each piece fits within
--- the token target ; then glues adjacent small pieces back up to
--- target, enforces a minimum-size floor on the trailing piece, and
--- prepends overlap from the preceding chunk.
---
--- Section-aware : when the input Doc carries `sections` from a
--- structured loader (markdown / html), each section is chunked
--- independently. Chunks never cross section boundaries.
---
--- The chunker takes `count_tokens` as a required injected callable, so
--- it stays tokenizer-agnostic. Wiring is the caller's responsibility :
--- ion7-core's Vocab tokenizer for production accuracy, a
--- 4-chars-per-token approximation for fast tests.

local M = {}

-- ── Separator hierarchy (coarse → fine) ─────────────────────────────────
--
-- Markdown headings are intentionally NOT in this list — section
-- segmentation is done by the loader, not by the chunker. The chunker
-- works inside an already-bounded section.

M.DEFAULT_SEPARATORS = {
    "\n\n",   -- paragraph
    "\n",     -- line
    ". ",     -- sentence (English / French)
    "? ",
    "! ",
    "; ",
    ", ",
    " ",      -- word
    "",       -- character (last resort)
}

-- ── Span-level helpers ──────────────────────────────────────────────────
--
-- A `span` is a half-open `{char_start, char_end}` table referencing
-- offsets in the original text. char_start is inclusive (0-based),
-- char_end is exclusive.

local function _span_text(text, span)
    return text:sub(span.char_start + 1, span.char_end)
end

--- Split `text[span]` on `sep`. Returns sub-spans referencing offsets
--- in `text`. Empty string sep means char-by-char.
local function _split_span(text, span, sep)
    local out = {}
    if sep == "" then
        for i = span.char_start, span.char_end - 1 do
            out[#out + 1] = { char_start = i, char_end = i + 1 }
        end
        return out
    end

    local pos = span.char_start
    while pos < span.char_end do
        local s, e = text:find(sep, pos + 1, true)
        if not s or s > span.char_end then
            out[#out + 1] = { char_start = pos, char_end = span.char_end }
            break
        end
        local zs = s - 1 -- 0-indexed start of separator match
        local ze = e     -- 0-indexed exclusive end of separator match
        if zs > pos then
            out[#out + 1] = { char_start = pos, char_end = zs }
        end
        pos = ze
        if pos >= span.char_end then break end
    end
    return out
end

-- ── Recursive split ─────────────────────────────────────────────────────

local function _recursive_split(text, span, target_tokens, count_tokens, seps, sep_idx)
    sep_idx = sep_idx or 1

    if count_tokens(_span_text(text, span)) <= target_tokens then
        return { span }
    end
    if sep_idx > #seps then
        return { span } -- can't split further ; oversize chunk
    end

    local sub = _split_span(text, span, seps[sep_idx])
    if #sub <= 1 then
        return _recursive_split(text, span, target_tokens, count_tokens, seps, sep_idx + 1)
    end

    local out = {}
    for _, p in ipairs(sub) do
        if count_tokens(_span_text(text, p)) <= target_tokens then
            out[#out + 1] = p
        else
            local further = _recursive_split(text, p, target_tokens, count_tokens, seps, sep_idx + 1)
            for _, f in ipairs(further) do out[#out + 1] = f end
        end
    end
    return out
end

-- ── Glue small pieces up to target_tokens ───────────────────────────────

local function _glue(text, pieces, target_tokens, count_tokens)
    local out = {}
    local cur
    for _, p in ipairs(pieces) do
        if not cur then
            cur = { char_start = p.char_start, char_end = p.char_end }
        else
            local merged_text = text:sub(cur.char_start + 1, p.char_end)
            if count_tokens(merged_text) <= target_tokens then
                cur.char_end = p.char_end
            else
                out[#out + 1] = cur
                cur = { char_start = p.char_start, char_end = p.char_end }
            end
        end
    end
    if cur then out[#out + 1] = cur end
    return out
end

-- ── Enforce min_tokens floor on the trailing chunk ──────────────────────

local function _merge_short_tail(text, glued, min_tokens, count_tokens)
    if #glued < 2 then return glued end
    local last = glued[#glued]
    local tail = text:sub(last.char_start + 1, last.char_end)
    if count_tokens(tail) < min_tokens then
        local prev = glued[#glued - 1]
        prev.char_end = last.char_end
        glued[#glued] = nil
    end
    return glued
end

-- ── Overlap (snap to word boundary) ─────────────────────────────────────
--
-- The overlap window is sized via a 4-chars-per-token heuristic, then
-- snapped to the nearest whitespace to avoid cutting mid-word. The
-- exact token count of the materialised chunk is recomputed at
-- emission time, so the heuristic only affects the overlap length, not
-- the reported `n_tokens`.

local function _overlap_start(text, prev_span, overlap_tokens)
    if overlap_tokens <= 0 then return nil end
    local est_chars = overlap_tokens * 4
    local start = math.max(prev_span.char_start, prev_span.char_end - est_chars)
    -- Snap forward to the next whitespace, then advance past it.
    while start < prev_span.char_end do
        local c = text:sub(start + 1, start + 1)
        if c == " " or c == "\n" or c == "\t" then break end
        start = start + 1
    end
    if start < prev_span.char_end then start = start + 1 end
    return start
end

-- ── Public entry point ──────────────────────────────────────────────────

--- @class Chunk
--- @field section    string?  Path from the source loader's section, if any.
--- @field char_start integer  0-based offset in Doc.text (inclusive).
--- @field char_end   integer  0-based offset in Doc.text (exclusive).
--- @field raw_text   string   Materialised chunk text (overlap included).
--- @field n_tokens   integer  Token count of raw_text.

--- Chunk a Doc into a list of Chunks ready for embedding and storage.
---
--- @param  doc  Doc       Loader output ; `text` and optional `sections`
---                  are read.
--- @param  opts table {
---     count_tokens   function(string) -> integer  REQUIRED. Tokenizer
---                  callable. Must be deterministic for stable boundaries.
---     target_tokens  integer  default 512. Soft cap for each chunk.
---     overlap_tokens integer  default 64. Number of tokens of the
---                  preceding chunk to prepend (snapped to word boundary).
---     min_tokens     integer  default 150. Floor below which a trailing
---                  chunk is merged back into its predecessor.
---     separators     string[] default DEFAULT_SEPARATORS. Hierarchy
---                  applied coarsest → finest.
--- }
--- @return Chunk[]  Chunks in document order ; each `char_start` /
---                  `char_end` references `doc.text`.
--- @raise When `opts.count_tokens` is missing or not callable.
function M.chunk(doc, opts)
    opts = opts or {}
    local count_tokens = opts.count_tokens
    if type(count_tokens) ~= "function" then
        error("ion7.rag.chunk.recursive : opts.count_tokens is required " ..
              "(callable string -> integer)", 2)
    end
    local target_tokens  = opts.target_tokens  or 512
    local overlap_tokens = opts.overlap_tokens or 64
    local min_tokens     = opts.min_tokens     or 150
    local separators     = opts.separators     or M.DEFAULT_SEPARATORS

    local text     = doc.text
    local sections = doc.sections
    if not sections or #sections == 0 then
        sections = { { path = "", char_start = 0, char_end = #text } }
    end

    local chunks = {}
    for _, section in ipairs(sections) do
        if section.char_end > section.char_start then
            local pieces = _recursive_split(text, section, target_tokens,
                                            count_tokens, separators, 1)
            local glued  = _glue(text, pieces, target_tokens, count_tokens)
            glued        = _merge_short_tail(text, glued, min_tokens, count_tokens)

            for i, p in ipairs(glued) do
                local effective_start = p.char_start
                if i > 1 then
                    local ov = _overlap_start(text, glued[i - 1], overlap_tokens)
                    if ov and ov < p.char_start then effective_start = ov end
                end
                local raw_text = text:sub(effective_start + 1, p.char_end)
                chunks[#chunks + 1] = {
                    section    = (section.path ~= "" and section.path) or nil,
                    char_start = effective_start,
                    char_end   = p.char_end,
                    raw_text   = raw_text,
                    n_tokens   = count_tokens(raw_text),
                }
            end
        end
    end

    return chunks
end

return M
