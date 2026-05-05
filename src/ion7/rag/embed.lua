--- @module ion7.rag.embed
--- @author  ion7 / Ion7 Project Contributors
---
--- Helpers that bind an externally-built `ion7.llm.Embed` instance to
--- the ion7-rag index tier. Two responsibilities :
---
---   - encode a query string into a query vector for `retrieve.search`
---   - bulk-encode chunk texts and upsert them into `idx.chunks_vec`
---
--- The embedder itself (model handle, context, pooling strategy) is
--- owned by `ion7.llm.Embed` ; this module never wraps a model.

local vec_db = require "ion7.rag.db.vec"

local M = {}

-- ── Query-time helpers ──────────────────────────────────────────────────

--- Encode a query string into a single embedding vector.
---
--- @param  embedder  ion7.llm.Embed
--- @param  text      string
--- @return number[]                   Vector of length `embedder`'s dim.
function M.encode_query(embedder, text)
    return embedder:encode(text)
end

-- ── Ingestion-time helpers ──────────────────────────────────────────────

--- Bulk-encode chunk texts and upsert the resulting vectors into
--- `idx.chunks_vec`. The embedder's `:encode_many` runs the per-text
--- forward passes ; the upsert is wrapped in a single transaction
--- (see `ion7.rag.db.vec.upsert_many`).
---
--- @param  handle       ion7.rag.db.Handle
--- @param  embedder     ion7.llm.Embed
--- @param  chunk_rows   table[]  `{ { chunk_id = integer, text = string }, ... }`
--- @param  on_progress  function(done, total)?  Called once after the
---                       upsert completes ; takes `(n_rows, n_rows)`.
--- @return integer                   Number of rows indexed.
--- @raise When any row is missing `chunk_id` or `text`.
function M.index_chunks(handle, embedder, chunk_rows, on_progress)
    if #chunk_rows == 0 then return 0 end

    local texts = {}
    for i, row in ipairs(chunk_rows) do
        assert(row.chunk_id and row.text,
            "embed.index_chunks[" .. i .. "] : missing chunk_id / text")
        texts[i] = row.text
    end

    local vectors = embedder:encode_many(texts)

    local rows = {}
    for i, row in ipairs(chunk_rows) do
        rows[i] = { chunk_id = row.chunk_id, embedding = vectors[i] }
    end
    vec_db.upsert_many(handle, rows)

    if on_progress then on_progress(#rows, #rows) end
    return #rows
end

--- Cosine similarity between two equal-length float arrays. Returns 0
--- when either input has zero magnitude.
---
--- @param  a  number[]
--- @param  b  number[]  Must be the same length as `a`.
--- @return number       In `[-1, 1]` for non-degenerate inputs.
function M.cosine(a, b)
    local dot, na, nb = 0, 0, 0
    for i = 1, #a do
        dot = dot + a[i] * b[i]
        na  = na  + a[i] * a[i]
        nb  = nb  + b[i] * b[i]
    end
    if na == 0 or nb == 0 then return 0 end
    return dot / (math.sqrt(na) * math.sqrt(nb))
end

return M
