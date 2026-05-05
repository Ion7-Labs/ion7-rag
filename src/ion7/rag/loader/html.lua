--- @module ion7.rag.loader.html
--- @author  ion7 / Ion7 Project Contributors
---
--- HTML loader. Backed by `gumbo` (HTML5 reference parser, Apache-2.0 ;
--- bundled gumbo-parser in the rock).
---
--- The DOM is walked once, accumulating visible text into a single
--- linear string and snapshotting heading positions along the way.
--- Sections are derived from the H1-H6 hierarchy in document order,
--- exactly like the markdown loader.
---
--- `Doc.text` holds the **extracted plain text**, not the raw HTML.
--- `char_start` / `char_end` therefore reference the clean view rather
--- than the source markup. The trade-off is deliberate : FTS5 indexes
--- meaningful tokens, the chunker splits on real word boundaries,
--- and citations point at human-readable content. Callers that need
--- raw-HTML byte mapping should keep the original input around and
--- snapshot it themselves.

local gumbo = require "gumbo"

local M = {}

-- Tags whose subtree is dropped entirely from the linear text :
-- non-content elements that BM25 / dense retrieval should not see.
local _SKIP_TAGS = {
    script = true, style = true, noscript = true, template = true,
    nav    = true, footer = true, aside = true,
}

-- Block-level tags that introduce a paragraph break in the output.
local _BLOCK_TAGS = {
    p = true, div = true, section = true, article = true, main = true,
    li = true, tr = true, table = true, blockquote = true, pre = true,
    figure = true, dl = true, dt = true, dd = true, hr = true, br = true,
}

local function _heading_level(tag)
    return tag == "h1" and 1 or tag == "h2" and 2 or tag == "h3" and 3
        or tag == "h4" and 4 or tag == "h5" and 5 or tag == "h6" and 6
        or nil
end

local function _normalize_ws(s)
    return (s or ""):gsub("%s+", " "):gsub("^%s+", ""):gsub("%s+$", "")
end

local function _build_path(stack, level)
    local parts = {}
    for L = 1, level do parts[L] = stack[L] or "?" end
    return table.concat(parts, " > ")
end

-- ── DOM walker ──────────────────────────────────────────────────────────

local _walk

_walk = function(node, state)
    if not node then return end

    -- Text node check : `nodeName == "#text"` is the most portable
    -- way to identify a text leaf across DOM-like libraries.
    if node.nodeName == "#text" then
        local t = node.data or ""
        state.parts[#state.parts + 1] = t
        state.pos = state.pos + #t
        return
    end

    local tag = node.localName
    if not tag then return end
    if _SKIP_TAGS[tag] then return end

    local h_level = _heading_level(tag)
    if h_level then
        local title = _normalize_ws(node.textContent)
        if title ~= "" then
            -- Heading position is recorded at the START of the heading
            -- content in the linear text, matching the markdown
            -- loader convention where the heading line opens its
            -- section.
            state.headings[#state.headings + 1] = {
                level      = h_level,
                title      = title,
                char_start = state.pos,
            }
            state.parts[#state.parts + 1] = title .. "\n\n"
            state.pos = state.pos + #title + 2
        end
        return
    end

    if node.childNodes then
        for _, child in ipairs(node.childNodes) do _walk(child, state) end
    end

    if _BLOCK_TAGS[tag] then
        state.parts[#state.parts + 1] = "\n"
        state.pos = state.pos + 1
    end
end

-- ── Loader ──────────────────────────────────────────────────────────────

--- Parse an HTML document into a Doc with heading-derived section
--- breakpoints over its extracted plain text.
---
--- @param  raw_html  string
--- @param  opts      table?  Standard loader opts ; `id`, `source_uri`,
---                  `title` are consulted (`title` falls back to the
---                  first H1, then to `<title>`).
--- @return Doc
--- @raise When `gumbo.parse` fails to parse the input.
function M.load(raw_html, opts)
    opts = opts or {}

    local doc = gumbo.parse(raw_html)
    if not doc then
        error("ion7.rag.loader.html : gumbo.parse returned nil", 2)
    end

    local state = { parts = {}, pos = 0, headings = {} }

    -- Body is the canonical content root. For HTML fragments without
    -- a <body>, fall back to the top-level document element.
    local root = doc.body or doc.documentElement
    if root then _walk(root, state) end

    local text = table.concat(state.parts)

    local sections = {}
    local stack    = {}
    local first_h1
    local title_from_doctitle = doc.title and _normalize_ws(doc.title) or nil
    if title_from_doctitle == "" then title_from_doctitle = nil end

    if #state.headings == 0 then
        sections[#sections + 1] = {
            path       = "",
            char_start = 0,
            char_end   = #text,
        }
    else
        -- Whitespace-only preludes are absorbed into the first
        -- heading's section so retrieval never has to reason about
        -- empty anonymous sections.
        local first_offset       = state.headings[1].char_start
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

        for i, h in ipairs(state.headings) do
            while #stack >= h.level do table.remove(stack) end
            stack[h.level] = h.title
            if h.level == 1 and not first_h1 then first_h1 = h.title end

            local start = (i == 1) and first_section_start or h.char_start
            local stop  = (state.headings[i + 1] or {}).char_start or #text
            sections[#sections + 1] = {
                path       = _build_path(stack, h.level),
                char_start = start,
                char_end   = stop,
            }
        end
    end

    return {
        id         = opts.id or "(unidentified-html)",
        format     = "html",
        source_uri = opts.source_uri,
        title      = opts.title or first_h1 or title_from_doctitle,
        text       = text,
        sections   = sections,
        meta       = nil,
    }
end

return M
