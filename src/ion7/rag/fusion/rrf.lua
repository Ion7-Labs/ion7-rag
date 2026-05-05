--- @module ion7.rag.fusion.rrf
--- @author  ion7 / Ion7 Project Contributors
---
--- Reciprocal Rank Fusion (Cormack et al., SIGIR 2009). Combines
--- candidate lists by their ranks alone, ignoring raw scores. Robust
--- to scale differences between dense distance and BM25 score.
---
---   score(d) = sum over sources S of  w[S] / (k + rank_S(d))
---
--- where rank starts at 1 (best). `k = 60` is the canonical choice ;
--- larger `k` softens the contribution of top ranks, smaller `k`
--- makes the top of each list dominate.
---
--- Output is `{chunk_id, score}` ordered by descending score (higher
--- is better). This convention differs from `db.vec` / `db.lex`
--- which return distances (lower is better) — fusion outputs scores
--- because the math is naturally on the score side.

local M = {}

M.DEFAULT_K = 60

--- Fuse per-source hit lists via RRF.
---
--- @param  hits_by_source  table<string, Hit[]>  Per-source ranked hits ;
---                                    each list ordered by ascending
---                                    distance.
--- @param  opts            table?  {
---     k        number?  default 60. Reciprocal-rank smoothing constant.
---     weights  table?   { source_name = weight }, default 1 each.
--- }
--- @return Hit[]  Fused hits sorted by descending score.
function M.fuse(hits_by_source, opts)
    opts = opts or {}
    local k       = opts.k or M.DEFAULT_K
    local weights = opts.weights or {}

    local scores = {}
    for source, hits in pairs(hits_by_source) do
        local w = weights[source] or 1
        for rank, hit in ipairs(hits) do
            local id = hit.chunk_id
            scores[id] = (scores[id] or 0) + w / (k + rank)
        end
    end

    local out = {}
    for id, s in pairs(scores) do
        out[#out + 1] = { chunk_id = id, score = s }
    end
    table.sort(out, function(a, b) return a.score > b.score end)
    return out
end

return M
