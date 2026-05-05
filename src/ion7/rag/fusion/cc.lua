--- @module ion7.rag.fusion.cc
--- @author  ion7 / Ion7 Project Contributors
---
--- Convex Combination. Min-max normalises each source's distances to
--- [0, 1] (inverted so 1 = best, 0 = worst), then takes a weighted
--- sum. Bruch et al. (TOIS 2023) reports CC outperforms RRF in-domain
--- when the caller has even a handful of training queries to tune the
--- weights. The per-source min-max is distribution-free, which keeps
--- the output stable as raw score scales drift between sources.
---
---   norm_S(d) = (max_S - distance_S(d)) / (max_S - min_S)   in [0, 1]
---   score(d) = sum over S of  w[S] * norm_S(d)
---
--- Hits absent from a given source contribute 0 to the sum, matching
--- RRF and DBSF.

local M = {}

--- Fuse per-source hit lists via min-max convex combination.
---
--- @param  hits_by_source  table<string, Hit[]>  Each Hit must carry
---                                    `distance`.
--- @param  opts            table?  {
---     weights  table?  { source_name = weight }, default 1 each.
--- }
--- @return Hit[]  Fused hits sorted by descending score.
function M.fuse(hits_by_source, opts)
    opts = opts or {}
    local weights = opts.weights or {}

    local sum = {}

    for source, hits in pairs(hits_by_source) do
        local n = #hits
        if n > 0 then
            local lo = math.huge
            local hi = -math.huge
            for _, h in ipairs(hits) do
                if h.distance < lo then lo = h.distance end
                if h.distance > hi then hi = h.distance end
            end
            local span = hi - lo
            local w = weights[source] or 1

            if span == 0 then
                -- Degenerate : all candidates tied. Give them all 1.0
                -- (everyone's "best"), letting them rank by other sources.
                for _, h in ipairs(hits) do
                    sum[h.chunk_id] = (sum[h.chunk_id] or 0) + w * 1.0
                end
            else
                for _, h in ipairs(hits) do
                    local norm = (hi - h.distance) / span
                    sum[h.chunk_id] = (sum[h.chunk_id] or 0) + w * norm
                end
            end
        end
    end

    local out = {}
    for id, s in pairs(sum) do
        out[#out + 1] = { chunk_id = id, score = s }
    end
    table.sort(out, function(a, b) return a.score > b.score end)
    return out
end

return M
