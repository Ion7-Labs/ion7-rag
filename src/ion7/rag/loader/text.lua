--- @module ion7.rag.loader.text
--- @author  ion7 / Ion7 Project Contributors
---
--- Plain-text loader. Used for `.txt` files, log fragments, and ad-hoc
--- strings that have no structural markup. The Doc it produces :
---
---   - `text` is the input verbatim.
---   - `sections` is empty ; the chunker treats the whole text as a
---     single anonymous section.
---   - `meta` is nil.

local M = {}

--- Build a `text`-format Doc from a string.
---
--- @param  text  string
--- @param  opts  table?  Standard loader opts. Only `id`,
---                  `source_uri`, and `title` are consulted ;
---                  format-specific keys are ignored.
--- @return Doc
function M.load(text, opts)
    opts = opts or {}
    return {
        id         = opts.id or "(unidentified-text)",
        format     = "text",
        source_uri = opts.source_uri,
        title      = opts.title,
        text       = text,
        sections   = {},
        meta       = nil,
    }
end

return M
