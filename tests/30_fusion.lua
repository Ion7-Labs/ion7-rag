#!/usr/bin/env luajit
--- @module tests.30_fusion
---
--- Synthetic-list tests for the three v1 fusion strategies.

local T = require "tests.framework"
require "tests.helpers"

local fusion = require "ion7.rag.fusion"
local rrf    = require "ion7.rag.fusion.rrf"
local dbsf   = require "ion7.rag.fusion.dbsf"
local cc     = require "ion7.rag.fusion.cc"

local function ranks(out)
    local r = {}
    for i, h in ipairs(out) do r[h.chunk_id] = i end
    return r
end

local function ids(out)
    local x = {}
    for i, h in ipairs(out) do x[i] = h.chunk_id end
    return x
end

-- ── RRF ────────────────────────────────────────────────────────────────

T.suite("ion7.rag.fusion.rrf — Reciprocal Rank Fusion")

T.test("single source : output preserves ranking", function()
    local out = rrf.fuse({
        dense = {
            { chunk_id = 10, distance = 0.1 },
            { chunk_id = 20, distance = 0.2 },
            { chunk_id = 30, distance = 0.3 },
        },
    })
    T.eq(out[1].chunk_id, 10)
    T.eq(out[2].chunk_id, 20)
    T.eq(out[3].chunk_id, 30)
    -- Score must be DESC (higher = better).
    T.gt(out[1].score, out[2].score)
    T.gt(out[2].score, out[3].score)
end)

T.test("two sources combine ranks", function()
    -- chunk 1 is top of dense, 3rd in lex.
    -- chunk 2 is bottom of dense, top of lex.
    -- chunk 3 is middle in both.
    -- chunk 4 only in dense.
    local out = rrf.fuse({
        dense = {
            { chunk_id = 1, distance = 0.1 },
            { chunk_id = 3, distance = 0.2 },
            { chunk_id = 2, distance = 0.5 },
            { chunk_id = 4, distance = 0.6 },
        },
        lex = {
            { chunk_id = 2, distance = -0.1 },
            { chunk_id = 3, distance = -0.5 },
            { chunk_id = 1, distance = -0.7 },
        },
    })
    -- All four should appear.
    local r = ranks(out)
    T.ok(r[1] and r[2] and r[3] and r[4])
    -- chunk 3 is solid mid in both -> often beats a single-source-top.
    -- We don't pin the exact order, just sanity-check that the
    -- in-both-lists chunks rank above the dense-only chunk 4.
    T.ok(r[3] < r[4])
end)

T.test("weights bias the contribution per source", function()
    local hits = {
        dense = {
            { chunk_id = 1, distance = 0 },
            { chunk_id = 2, distance = 0 },
        },
        lex = {
            { chunk_id = 2, distance = 0 },
            { chunk_id = 1, distance = 0 },
        },
    }
    local out_dense_heavy = rrf.fuse(hits, { weights = { dense = 4, lex = 1 } })
    local out_lex_heavy   = rrf.fuse(hits, { weights = { dense = 1, lex = 4 } })

    T.eq(out_dense_heavy[1].chunk_id, 1, "dense-heavy must promote chunk 1")
    T.eq(out_lex_heavy  [1].chunk_id, 2, "lex-heavy must promote chunk 2")
end)

T.test("k constant softens / sharpens top-rank dominance", function()
    -- With very small k, top-1 of any source dominates.
    local hits = {
        dense = { { chunk_id = 1, distance = 0 }, { chunk_id = 2, distance = 0 } },
        lex   = { { chunk_id = 3, distance = 0 }, { chunk_id = 4, distance = 0 } },
    }
    local out_small_k = rrf.fuse(hits, { k = 1 })
    -- k=1 -> rank-1 contributes 1/(1+1)=0.5, rank-2 contributes 1/3=0.33.
    -- chunk 1 and chunk 3 are both rank-1 in their source ; tied for top.
    T.ok(out_small_k[1].chunk_id == 1 or out_small_k[1].chunk_id == 3)

    -- With k=1000, ranks barely differ ; still all four present.
    local out_big_k = rrf.fuse(hits, { k = 1000 })
    T.eq(#out_big_k, 4)
end)

T.test("registry dispatches to rrf", function()
    local out = fusion.fuse("rrf", {
        dense = { { chunk_id = 7, distance = 0 } },
    })
    T.eq(out[1].chunk_id, 7)
end)

-- ── DBSF ───────────────────────────────────────────────────────────────

T.suite("ion7.rag.fusion.dbsf — Distribution-Based Score Fusion")

T.test("single source : closer distance scores higher", function()
    local out = dbsf.fuse({
        dense = {
            { chunk_id = 1, distance = 0.1 },
            { chunk_id = 2, distance = 0.5 },
            { chunk_id = 3, distance = 0.9 },
        },
    })
    T.eq(out[1].chunk_id, 1)
    T.eq(out[3].chunk_id, 3)
    T.gt(out[1].score, out[3].score)
end)

T.test("two sources sum z-scores", function()
    -- chunk 5 is the best dense AND the best lex. Should top.
    local out = dbsf.fuse({
        dense = {
            { chunk_id = 5, distance = 0.0 },
            { chunk_id = 6, distance = 0.5 },
            { chunk_id = 7, distance = 1.0 },
        },
        lex = {
            { chunk_id = 5, distance = -10 },
            { chunk_id = 6, distance = -5  },
            { chunk_id = 7, distance =  0  },
        },
    })
    T.eq(out[1].chunk_id, 5)
end)

T.test("identical distances within a source contribute z = 0", function()
    -- The "lex" source is fully degenerate ; only "dense" decides.
    local out = dbsf.fuse({
        dense = {
            { chunk_id = 1, distance = 0.0 },
            { chunk_id = 2, distance = 1.0 },
        },
        lex = {
            { chunk_id = 1, distance = 7.0 },
            { chunk_id = 2, distance = 7.0 },
            { chunk_id = 3, distance = 7.0 },
        },
    })
    T.eq(out[1].chunk_id, 1, "dense should decide when lex is degenerate")
end)

T.test("registry dispatches to dbsf", function()
    -- DBSF needs at least two distinct distances to produce a non-zero
    -- standard deviation ; a single-element source is degenerate and
    -- correctly contributes nothing. Use two hits with different
    -- distances so we exercise the dispatch path on real output.
    local out = fusion.fuse("dbsf", {
        dense = {
            { chunk_id = 9,  distance = 0.1 },
            { chunk_id = 10, distance = 0.9 },
        },
    })
    T.eq(out[1].chunk_id, 9)
    T.eq(out[2].chunk_id, 10)
end)

-- ── CC ─────────────────────────────────────────────────────────────────

T.suite("ion7.rag.fusion.cc — Convex Combination")

T.test("single source : min-max preserves order", function()
    local out = cc.fuse({
        dense = {
            { chunk_id = 1, distance = 0.0 },
            { chunk_id = 2, distance = 0.5 },
            { chunk_id = 3, distance = 1.0 },
        },
    })
    T.eq(out[1].chunk_id, 1)
    T.eq(out[3].chunk_id, 3)
    -- Scores must be in [0, 1] for a single source.
    T.eq(out[1].score, 1.0)
    T.eq(out[3].score, 0.0)
end)

T.test("weights tilt the combination", function()
    local hits = {
        dense = { { chunk_id = 1, distance = 0 }, { chunk_id = 2, distance = 1 } },
        lex   = { { chunk_id = 2, distance = -10 }, { chunk_id = 1, distance = 0 } },
    }
    -- Equal weights : each source contributes [0,1] equally ; tie at sum 1.0.
    -- Dense-heavy : chunk 1 wins (best dense, worst lex but lex weighted low).
    local dense_heavy = cc.fuse(hits, { weights = { dense = 4, lex = 1 } })
    T.eq(dense_heavy[1].chunk_id, 1)

    local lex_heavy = cc.fuse(hits, { weights = { dense = 1, lex = 4 } })
    T.eq(lex_heavy[1].chunk_id, 2)
end)

T.test("tied source contributes 1.0 per hit (degenerate handling)", function()
    -- Single-element source : span = 0 ; we treat all hits as tied at top.
    local out = cc.fuse({ dense = { { chunk_id = 7, distance = 99 } } })
    T.eq(out[1].chunk_id, 7)
    T.eq(out[1].score,    1.0)
end)

T.test("registry dispatches to cc", function()
    local out = fusion.fuse("cc", { dense = { { chunk_id = 11, distance = 0 } } })
    T.eq(out[1].chunk_id, 11)
end)

-- ── Registry errors ────────────────────────────────────────────────────

T.suite("ion7.rag.fusion — registry")

T.test("unknown strategy raises clearly", function()
    T.err(function() fusion.fuse("not-a-real-strategy", {}) end, "unknown strategy")
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
