--- @module ion7.rag.loader
--- @author  ion7 / Ion7 Project Contributors
---
--- Loader registry and format dispatch. Every per-format loader
--- (`loader.text`, `loader.markdown`, `loader.html`, ...) returns the
--- same `Doc` shape so downstream chunkers can stay format-agnostic.
---
---   Doc = {
---       id          string     caller-provided stable identifier
---       format      string     "text" | "markdown" | "html" | ...
---       source_uri  string?    file path / URL / nil for in-memory
---       title       string?
---       text        string     full original text ; section offsets
---                                refer to this string
---       sections    Section[]  ordered, non-overlapping, may be empty
---       meta        table?     format-specific metadata blob
---   }
---
---   Section = {
---       path        string     "Header A > Header B" hierarchy
---       char_start  integer    inclusive byte offset in Doc.text
---       char_end    integer    exclusive byte offset in Doc.text
---   }
---
--- Empty `sections` is a valid Doc ; the chunker treats the whole
--- text as a single anonymous section in that case.

local M = {}

-- ── Format detection ────────────────────────────────────────────────────

local _EXT_TO_FORMAT = {
    txt      = "text",
    text     = "text",
    md       = "markdown",
    markdown = "markdown",
    html     = "html",
    htm      = "html",
}

--- Infer a loader format from a filename extension.
---
--- @param  path  string
--- @return string?  Format name (`"text"`, `"markdown"`, `"html"`),
---                  or nil when the extension is unrecognised. In the
---                  nil case the caller must pass `opts.format` to
---                  `from_file` explicitly.
function M.detect_format(path)
    local ext = path:match("^.*%.([%w]+)$")
    if not ext then return nil end
    return _EXT_TO_FORMAT[ext:lower()]
end

-- ── Loader registry ─────────────────────────────────────────────────────

local _LOADERS = {
    text     = "ion7.rag.loader.text",
    markdown = "ion7.rag.loader.markdown",
    html     = "ion7.rag.loader.html",
}

--- Resolve a loader module by format name.
---
--- @param  format  string  `"text"` | `"markdown"` | `"html"`.
--- @return table   The loader module (exposes `load(text, opts)`).
--- @raise When the format name is not registered.
function M.for_format(format)
    local mod = _LOADERS[format]
    if not mod then
        error("ion7.rag.loader : unknown format '" .. tostring(format) ..
              "' (registered : text, markdown, html)", 2)
    end
    return require(mod)
end

--- Build a `Doc` from an in-memory string.
---
--- @param  text  string
--- @param  opts  table {
---     format      string   REQUIRED. `"text"` | `"markdown"` | `"html"`.
---     id          string?  Caller-provided stable identifier. Loaders
---                  fall back to a format-specific placeholder when nil.
---     source_uri  string?
---     title       string?
---     ...  Format-specific options forwarded verbatim to the loader.
--- }
--- @return Doc
--- @raise When `opts.format` is missing or unrecognised.
function M.from_string(text, opts)
    opts = opts or {}
    assert(opts.format, "ion7.rag.loader.from_string : opts.format required")
    local mod = M.for_format(opts.format)
    return mod.load(text, opts)
end

--- Build a `Doc` from a file on disk. The format is inferred from the
--- extension when `opts.format` is not provided. `opts.source_uri`
--- defaults to `path`, and `opts.id` defaults to `path` so consecutive
--- ingests of the same file produce stable doc identifiers.
---
--- @param  path  string
--- @param  opts  table?  Same shape as `from_string`'s opts ; `format`
---                  is optional here because of extension inference.
--- @return Doc
--- @raise When the format cannot be inferred and is not provided, or
---        when the file cannot be opened.
function M.from_file(path, opts)
    opts = opts or {}
    if not opts.format then
        opts.format = M.detect_format(path)
        if not opts.format then
            error("ion7.rag.loader.from_file : cannot infer format from '" ..
                  path .. "' — pass opts.format explicitly", 2)
        end
    end
    local f, err = io.open(path, "rb")
    if not f then
        error("ion7.rag.loader.from_file : " .. tostring(err), 2)
    end
    local text = f:read("*a")
    f:close()

    opts.source_uri = opts.source_uri or path
    if not opts.id then opts.id = path end
    return M.from_string(text, opts)
end

return M
