--- @module ion7.rag.retrieve
--- @author  ion7 / Ion7 Project Contributors
---
--- Hybrid retrieval glue : pull candidates from the dense tier
--- (sqlite-vec), the lexical tier (FTS5 BM25), and optionally the HyPE
--- tier (`idx.hype_vec` collapsed to one entry per chunk_id) ; then
--- fuse the resulting ranked lists with the configured strategy
--- (RRF / DBSF / CC). Returns `{chunk_id, score}` hits ordered by
--- descending score ; the caller hydrates the chunk text from the
--- truth tier when needed.
---
--- Recommended sizing : `k_dense = k_lex ≈ 5–10 × k_final`, fuse with
--- RRF as a zero-shot default, then rerank the top of the fused list
--- with a cross-encoder. The reranker step is owned by the calling
--- `Pipeline` ; this module is the candidate-fetch layer beneath it.

local vec    = require "ion7.rag.db.vec"
local lex    = require "ion7.rag.db.lex"
local hype   = require "ion7.rag.db.hype"
local fusion = require "ion7.rag.fusion"

local M = {}

-- ── Defaults ────────────────────────────────────────────────────────────

M.DEFAULTS = {
    k_dense  = 50,
    k_lex    = 50,
    k_hype   = 50,
    k_final  = 10,
    fusion   = "rrf",
    -- "full" = fp32 vectors (high precision, slower). "binary" reads
    -- the MRL-truncated binary tier and runs Hamming distance — much
    -- faster, good for shortlist scenarios.
    vec_tier = "full",
    -- "auto" enables HyPE folding only when `idx.hype_vec` contains rows
    -- (i.e. when the caller paid for HyPE generation at ingest time).
    use_hype = "auto",
}

-- ── search ──────────────────────────────────────────────────────────────

--- Return true when `idx.hype_vec` contains at least one row. Used by
--- the `use_hype = "auto"` default. One COUNT query per retrieval.
local function _has_hype(h)
    local stmt = h:prepare("SELECT COUNT(*) FROM idx.hype_vec")
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n > 0
end

--- Run a hybrid retrieval and return a fused ranked list of hits.
--- At least one of `query_vec` / `query_text` must be provided.
---
--- @param  h     ion7.rag.db.Handle
--- @param  opts  table {
---     query_vec   number[]?     Embedding query vector. Length must
---                  equal `h:embed_dim()`.
---     query_text  string?       FTS5 query expression.
---     k_dense     integer?      Default 50.
---     k_lex       integer?      Default 50.
---     k_hype      integer?      Default 50. Raw HyPE pull, before
---                  collapse-by-chunk-id.
---     k_final     integer?      Default 10. Truncates the fused list.
---     fusion      string?       "rrf" (default) | "dbsf" | "cc".
---     fusion_k    number?       RRF-only constant. Default 60.
---     weights     table?        Per-source fusion weights, e.g.
---                  `{ dense = 4, lex = 1, hype = 4 }`.
---     vec_tier    string?       "full" (default) | "binary".
---     use_hype    bool|"auto"?  Default "auto" ; enables HyPE folding
---                  only when the index actually has HyPE rows.
--- }
--- @return Hit[]  `{ { chunk_id, score }, ... }` ordered by descending score.
--- @raise When neither `query_vec` nor `query_text` is provided.
function M.search(h, opts)
    opts = opts or {}
    if not opts.query_vec and not opts.query_text then
        error("ion7.rag.retrieve.search : at least one of " ..
              "opts.query_vec or opts.query_text is required", 2)
    end

    local k_dense  = opts.k_dense  or M.DEFAULTS.k_dense
    local k_lex    = opts.k_lex    or M.DEFAULTS.k_lex
    local k_hype   = opts.k_hype   or M.DEFAULTS.k_hype
    local k_final  = opts.k_final  or M.DEFAULTS.k_final
    local strategy = opts.fusion   or M.DEFAULTS.fusion
    local vec_tier = opts.vec_tier or M.DEFAULTS.vec_tier

    local use_hype = opts.use_hype
    if use_hype == nil then use_hype = M.DEFAULTS.use_hype end
    if use_hype == "auto" then use_hype = _has_hype(h) end

    local hits_by_source = {}

    if opts.query_vec then
        local fn = (vec_tier == "binary") and vec.knn_binary or vec.knn_full
        hits_by_source.dense = fn(h, opts.query_vec, k_dense)

        if use_hype then
            local hype_fn = (vec_tier == "binary") and hype.knn_binary or hype.knn_full
            local raw = hype_fn(h, opts.query_vec, k_hype)
            hits_by_source.hype = hype.collapse_to_chunks(raw)
        end
    end

    if opts.query_text then
        hits_by_source.lex = lex.search(h, opts.query_text, k_lex)
    end

    -- Short-circuit when only one source contributed candidates :
    -- fusion on a single ranked list is a no-op pass-through, and we
    -- can hand back hits with an inverted-distance score so the
    -- caller-side score-descending convention still holds.
    local n_sources = 0
    for _ in pairs(hits_by_source) do n_sources = n_sources + 1 end
    if n_sources == 1 then
        local single
        for _, hits in pairs(hits_by_source) do single = hits end
        local out = {}
        for i = 1, math.min(#single, k_final) do
            out[i] = { chunk_id = single[i].chunk_id, score = -single[i].distance }
        end
        return out
    end

    local fused = fusion.fuse(strategy, hits_by_source, {
        weights = opts.weights,
        k       = opts.fusion_k,
    })

    local out = {}
    for i = 1, math.min(#fused, k_final) do out[i] = fused[i] end
    return out
end

return M
