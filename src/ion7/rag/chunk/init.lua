--- @module ion7.rag.chunk
--- @author  ion7 / Ion7 Project Contributors
---
--- Chunker registry. The recursive splitter is the default ; the late
--- chunker is also registered here for namespace discoverability,
--- though its public API (`encode`) does not match the chunker shape
--- expected by `M.chunk` and is consumed via
--- `ion7.rag.chunk.late.encode` directly.

local M = {}

local _CHUNKERS = {
    recursive = "ion7.rag.chunk.recursive",
    late      = "ion7.rag.chunk.late",
}

--- Resolve a chunker module by name.
---
--- @param  name  string  `"recursive"` (or `"late"`, see module
---                  doc above).
--- @return table         The chunker module.
--- @raise When the name is not registered.
function M.for_name(name)
    local mod = _CHUNKERS[name]
    if not mod then
        error("ion7.rag.chunk : unknown chunker '" .. tostring(name) ..
              "' (registered : recursive, late)", 2)
    end
    return require(mod)
end

--- Pass-through to the named chunker's `chunk(doc, opts)` function.
--- Only meaningful for chunkers that follow the recursive shape ;
--- `late` exposes `encode(opts)` instead and should be called directly.
---
--- @param  name  string
--- @param  doc   Doc
--- @param  opts  table?
--- @return Chunk[]
function M.chunk(name, doc, opts)
    return M.for_name(name).chunk(doc, opts)
end

return M
