#!/usr/bin/env luajit
--- @module tests.71_route_tfidf
---
--- TF-IDF + centroid query router. Pure-Lua, no model needed.

local T = require "tests.framework"
require "tests.helpers"

T.suite("ion7.rag.route.tfidf — TF-IDF + centroid classifier")

local Router = require("ion7.rag.route.tfidf").Router

-- ── Training set ───────────────────────────────────────────────────────
--
-- Three labels covering the typical agentic-RAG triage : skip the
-- retrieve, do a single retrieval round, or retrieve+synthesise across
-- multiple rounds.

local TRAIN = {
    -- no_retrieve : chit-chat, math, translation, code generation
    { query = "Hi there, how are you doing today?",                  label = "no_retrieve" },
    { query = "Tell me a short joke about cats",                      label = "no_retrieve" },
    { query = "What is 17 times 23?",                                 label = "no_retrieve" },
    { query = "Translate 'good morning' into Japanese",               label = "no_retrieve" },
    { query = "Write a Python function that reverses a list",         label = "no_retrieve" },
    { query = "Compute the cube root of 64",                          label = "no_retrieve" },

    -- single_hop : factoid, one chunk should answer
    { query = "When was the company founded?",                        label = "single_hop"  },
    { query = "What is the address of the head office?",              label = "single_hop"  },
    { query = "How long does standard delivery take?",                label = "single_hop"  },
    { query = "What is the cancellation policy?",                     label = "single_hop"  },
    { query = "Who is the current CEO of the firm?",                  label = "single_hop"  },
    { query = "What is the current price of the standard plan?",      label = "single_hop"  },

    -- multi_hop : compares / synthesises across docs
    { query = "Compare the delivery times across our two contracts",  label = "multi_hop"   },
    { query = "Summarise the differences between policy A and B",     label = "multi_hop"   },
    { query = "How did our refund policy change between 2024 and 2026?", label = "multi_hop" },
    { query = "Reconcile the discount tiers across both supplier agreements", label = "multi_hop" },
    { query = "What changed in the contract between renewals?",       label = "multi_hop"   },
}

-- ── Tests ──────────────────────────────────────────────────────────────

T.test("Router.fit builds a classifier with N labels and a vocab", function()
    local r = Router.fit(TRAIN)
    T.eq(r:n_labels(), 3)
    T.gt(r:vocab_size(), 30)
end)

T.test("classify routes chit-chat to no_retrieve", function()
    local r = Router.fit(TRAIN)
    local label = r:classify("Hi! Can you tell me a quick joke?")
    T.eq(label, "no_retrieve")
end)

T.test("classify routes a factoid to single_hop", function()
    local r = Router.fit(TRAIN)
    local label = r:classify("How long is standard shipping?")
    T.eq(label, "single_hop")
end)

T.test("classify routes a compare-style query to multi_hop", function()
    local r = Router.fit(TRAIN)
    local label = r:classify("Compare the discount tiers in the two agreements")
    T.eq(label, "multi_hop")
end)

T.test("classify returns a confidence score in [0, 1]", function()
    local r = Router.fit(TRAIN)
    local _, score, all = r:classify("How long does delivery take?")
    T.gte(score, 0)
    T.ok(score <= 1.0001, "score should not exceed 1 (got " .. score .. ")")
    T.is_type(all, "table")
    T.eq(all["single_hop"] ~= nil, true)
end)

T.test("classify on an empty / no-overlap query returns score 0", function()
    local r = Router.fit(TRAIN)
    local _, score = r:classify("zzzqqqxxx")
    T.eq(score, 0)
end)

T.test("Router.fit rejects an empty examples set", function()
    T.err(function() Router.fit({}) end, "no examples")
end)

T.test("classify is stable run-to-run on the same input", function()
    local r = Router.fit(TRAIN)
    local label_1 = r:classify("How long does standard shipping take?")
    local label_2 = r:classify("How long does standard shipping take?")
    T.eq(label_1, label_2)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
