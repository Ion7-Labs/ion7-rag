--- @module ion7.rag.loader.markdown
--- @author  ion7 / Ion7 Project Contributors
---
--- Markdown loader. Backed by `cmark` (CommonMark, BSD-2 ; bundled
--- libcmark in the rock, so no system dep beyond a C compiler).
---
--- The parser is invoked with `OPT_SOURCEPOS` so every AST node
--- carries its source line range. The top-level AST is walked once
--- to collect headings, and 1-based line positions are translated
--- back to 0-based byte offsets into the original text. Sections
--- span from one heading line to the line of the next.
---
---   Doc.text       Raw markdown, byte-identical to the input. Section
---                  `char_start` / `char_end` index into this string.
---   Doc.sections   Ordered, non-overlapping spans (one per heading,
---                  plus an optional anonymous prelude).
---   Doc.title      First H1's plain-text content, when present.

local cmark = require "cmark"

local M = {}

-- ── Helpers ─────────────────────────────────────────────────────────────

--- Build a per-line byte-offset table : `offsets[L]` is the 0-based
--- byte offset where line L starts. Lines are 1-indexed. The sentinel
--- entry `offsets[#lines + 1]` points to one past the end of the text,
--- letting "line just after the last" lookups work uniformly.
local function _line_offsets(text)
    local offsets = { 0 }
    local pos = 1
    while true do
        local s = text:find("\n", pos, true)
        if not s then break end
        offsets[#offsets + 1] = s -- start of line N+1, 0-indexed
        pos = s + 1
    end
    offsets[#offsets + 1] = #text
    return offsets
end

--- Recursively concatenate the literal text of a node's children. Used
--- to recover the rendered title of a heading without round-tripping
--- through `cmark.render_commonmark` (which would re-emit `# ...`).
local function _heading_text(node)
    local out = {}
    local c = cmark.node_first_child(node)
    while c do
        local t = cmark.node_get_type_string(c)
        if t == "text" or t == "code" then
            out[#out + 1] = cmark.node_get_literal(c) or ""
        elseif t == "softbreak" or t == "linebreak" then
            out[#out + 1] = " "
        else
            -- Recurse into emph / strong / link / etc.
            out[#out + 1] = _heading_text(c)
        end
        c = cmark.node_next(c)
    end
    return table.concat(out)
end

local function _build_path(stack, level)
    local parts = {}
    for L = 1, level do parts[L] = stack[L] or "?" end
    return table.concat(parts, " > ")
end

-- ── Loader ──────────────────────────────────────────────────────────────

--- Parse a Markdown document into a Doc with heading-derived section
--- breakpoints.
---
--- @param  text  string  Source markdown.
--- @param  opts  table?  Standard loader opts ; `id`, `source_uri`,
---                  `title` are consulted (`title` falls back to the
---                  first H1 when present).
--- @return Doc
--- @raise When `cmark.parse_string` fails to parse the input.
function M.load(text, opts)
    opts = opts or {}

    local root = cmark.parse_string(text, cmark.OPT_SOURCEPOS)
    if not root then
        error("ion7.rag.loader.markdown : cmark.parse_string returned nil", 2)
    end

    local offsets = _line_offsets(text)
    local headings = {}

    local n = cmark.node_first_child(root)
    while n do
        if cmark.node_get_type_string(n) == "heading" then
            headings[#headings + 1] = {
                level = cmark.node_get_heading_level(n),
                line  = cmark.node_get_start_line(n),
                title = _heading_text(n):gsub("%s+$", ""),
            }
        end
        n = cmark.node_next(n)
    end

    local sections = {}
    local stack    = {}
    local first_h1

    if #headings == 0 then
        -- No structure : one anonymous root section covering the whole
        -- document. The chunker will treat it as a single span.
        sections[#sections + 1] = {
            path       = "",
            char_start = 0,
            char_end   = #text,
        }
    else
        -- Optional prelude before the first heading. Absorb whitespace-
        -- only preludes into the first heading's section so retrieval
        -- never has to deal with empty / blank-only anonymous sections.
        local first_offset      = offsets[headings[1].line] or #text
        local first_section_start = first_offset
        if first_offset > 0 then
            local prelude = text:sub(1, first_offset)
            if prelude:find("%S") then
                sections[#sections + 1] = {
                    path       = "",
                    char_start = 0,
                    char_end   = first_offset,
                }
            else
                first_section_start = 0
            end
        end

        for i, h in ipairs(headings) do
            while #stack >= h.level do table.remove(stack) end
            stack[h.level] = h.title
            if h.level == 1 and not first_h1 then first_h1 = h.title end

            local start = offsets[h.line] or #text
            if i == 1 then start = first_section_start end
            local stop  = offsets[(headings[i + 1] or {}).line or (#offsets)] or #text

            sections[#sections + 1] = {
                path       = _build_path(stack, h.level),
                char_start = start,
                char_end   = stop,
            }
        end
    end

    return {
        id         = opts.id or "(unidentified-markdown)",
        format     = "markdown",
        source_uri = opts.source_uri,
        title      = opts.title or first_h1,
        text       = text,
        sections   = sections,
        meta       = nil,
    }
end

--- Render a markdown string to plain text. Strips inline and block
--- markup while preserving readable structure (paragraph breaks,
--- list-item separation). Suitable for FTS5 indexing or for handing a
--- chunk to a model that doesn't need to see the markup.
---
--- @param  s  string  Markdown source.
--- @return string     Plain-text rendering, or `s` itself when
---                     `cmark.parse_string` fails.
function M.to_plain_text(s)
    local node = cmark.parse_string(s, cmark.OPT_DEFAULT)
    if not node then return s end
    -- Some cmark-lua builds omit `render_plaintext` ; the manual walk
    -- below covers that case identically.
    local ok, out = pcall(cmark.render_plaintext, node, cmark.OPT_DEFAULT, 0)
    if ok and out then return out end

    local function walk(n)
        local acc = {}
        local c = cmark.node_first_child(n)
        while c do
            local t = cmark.node_get_type_string(c)
            if t == "text" or t == "code" then
                acc[#acc + 1] = cmark.node_get_literal(c) or ""
            elseif t == "softbreak" then
                acc[#acc + 1] = " "
            elseif t == "linebreak" or t == "paragraph" or t == "heading"
                or t == "list" or t == "item" or t == "block_quote" then
                acc[#acc + 1] = walk(c)
                acc[#acc + 1] = "\n"
            else
                acc[#acc + 1] = walk(c)
            end
            c = cmark.node_next(c)
        end
        return table.concat(acc)
    end
    return walk(node)
end

return M
