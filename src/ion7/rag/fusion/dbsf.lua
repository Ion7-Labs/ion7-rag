--- @module ion7.rag.fusion.dbsf
--- @author  ion7 / Ion7 Project Contributors
---
--- Distribution-Based Score Fusion. For each source, compute the mean
--- and standard deviation of the candidate distances, z-score every
--- hit (inverted so that smaller distance maps to larger z), and sum
--- the z-scores across sources. Bruch et al., "Analysis of Fusion
--- Functions for Hybrid Retrieval" (TOIS 2023, arXiv:2210.11934)
--- reports DBSF outperforms RRF when per-source distance
--- distributions are well-behaved — query-adaptive without training
--- data.
---
---   z_S(d)   = (mean_S - distance_S(d)) / std_S
---   score(d) = sum over S of  w[S] * z_S(d)
---
--- A source whose distances are all identical yields std = 0 and is
--- skipped for that query (every hit would z-score to 0, dropping
--- the source out without dividing by zero).

local M = {}

--- Fuse per-source hit lists via DBSF.
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
            local mean = 0
            for _, h in ipairs(hits) do mean = mean + h.distance end
            mean = mean / n

            local sumsq = 0
            for _, h in ipairs(hits) do
                local d = h.distance - mean
                sumsq = sumsq + d * d
            end
            local sd = math.sqrt(sumsq / n)

            local w = weights[source] or 1
            if sd == 0 then
                -- All distances identical : every hit would z-score
                -- to 0 and the source contributes nothing. Skipping
                -- the loop avoids a div-by-zero.
            else
                for _, h in ipairs(hits) do
                    local z = (mean - h.distance) / sd
                    sum[h.chunk_id] = (sum[h.chunk_id] or 0) + w * z
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
