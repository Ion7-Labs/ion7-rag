--- @module ion7.rag.route.tfidf
--- @author  ion7 / Ion7 Project Contributors
---
--- TF-IDF + per-label centroid classifier (RAGRouter-Bench baseline).
--- Label-agnostic : the caller passes labelled examples to `Router.fit`
--- and the classifier picks them up. The pipeline default labels are
--- `no_retrieve`, `single_hop`, `multi_hop`, but any label set works.
---
--- The fitted Router holds an IDF table (one float per vocab term) and
--- a centroid per label (sparse `term → tfidf_avg` map plus its L2
--- norm). Classification at query time : tokenise, term-frequency,
--- multiply by stored IDF, cosine against each centroid, argmax.
---
--- All math runs on plain Lua tables — no external dependencies, no
--- BLAS, no FFI. Routing decision is sub-millisecond per query.

local M = {}

-- ── Tokenisation ────────────────────────────────────────────────────────
--
-- Crude unicode-friendly word split : lowercase + split on any non-
-- alphanumeric. Latin and basic Cyrillic survive ; CJK collapses into
-- one big token per script run, which underweights but does not break
-- routing — the routing decision is a coarse three-way pick, not a
-- per-token quality gate.

local function _tokenise(s)
    if not s or s == "" then return {} end
    local out = {}
    for tok in s:lower():gmatch("[%w']+") do
        out[#out + 1] = tok
    end
    return out
end

local function _term_freq(tokens)
    local n = #tokens
    if n == 0 then return {}, 0 end
    local tf = {}
    for _, t in ipairs(tokens) do tf[t] = (tf[t] or 0) + 1 end
    -- Normalise by total length so long queries do not blow scores.
    for t, c in pairs(tf) do tf[t] = c / n end
    return tf, n
end

local function _norm(vec)
    local s = 0
    for _, v in pairs(vec) do s = s + v * v end
    return math.sqrt(s)
end

-- ── Class ───────────────────────────────────────────────────────────────

local Router = {}
Router.__index = Router
M.Router = Router

--- Fit a router from labelled examples.
---
--- @param  examples table[]  { { query = string, label = string }, ... }
--- @return Router
function Router.fit(examples)
    assert(#examples > 0, "Router.fit : no examples")

    local docs_per_term = {}
    local docs = {}
    local labels_set = {}

    for i, ex in ipairs(examples) do
        assert(ex.query and ex.label,
            "Router.fit[" .. i .. "] : missing query / label")
        local tokens = _tokenise(ex.query)
        local seen = {}
        for _, t in ipairs(tokens) do seen[t] = true end
        for t in pairs(seen) do
            docs_per_term[t] = (docs_per_term[t] or 0) + 1
        end
        docs[i] = { tokens = tokens, label = ex.label }
        labels_set[ex.label] = true
    end

    local n_docs = #docs
    local idf = {}
    for t, df in pairs(docs_per_term) do
        idf[t] = math.log((1 + n_docs) / (1 + df)) + 1
    end

    -- Build per-label TF-IDF averages.
    local sums   = {}  -- label -> { term -> sum_tfidf }
    local counts = {}  -- label -> count
    for _, d in ipairs(docs) do
        local tf = _term_freq(d.tokens)
        local s = sums[d.label] or {}
        for term, v in pairs(tf) do
            s[term] = (s[term] or 0) + v * (idf[term] or 0)
        end
        sums[d.label] = s
        counts[d.label] = (counts[d.label] or 0) + 1
    end

    local centroids = {}
    local norms     = {}
    local labels    = {}
    for label, s in pairs(sums) do
        local c = {}
        for term, total in pairs(s) do
            c[term] = total / counts[label]
        end
        centroids[label] = c
        norms[label]     = _norm(c)
        labels[#labels + 1] = label
    end
    table.sort(labels)

    return setmetatable({
        _idf       = idf,
        _centroids = centroids,
        _norms     = norms,
        _labels    = labels,
    }, Router)
end

--- Classify a query.
---
--- @param  query  string
--- @return string  label   Best-matching label (alphabetically first
---                  on ties).
--- @return number  score   Cosine in [0, 1] between query and the
---                  winning centroid. 0 when the query shares no terms
---                  with any centroid — callers can treat that as
---                  "uncertain".
--- @return table   scores  Full distribution `label → cosine`.
function Router:classify(query)
    local tokens = _tokenise(query)
    local tf = _term_freq(tokens)
    local q = {}
    for term, v in pairs(tf) do
        local i = self._idf[term]
        if i then q[term] = v * i end
    end
    local q_norm = _norm(q)

    local best_label = self._labels[1]
    local best_score = 0
    local scores     = {}
    for _, label in ipairs(self._labels) do
        local s = 0
        if q_norm > 0 then
            local centroid = self._centroids[label]
            local dot = 0
            for term, v in pairs(q) do
                local w = centroid[term]
                if w then dot = dot + v * w end
            end
            local cn = self._norms[label]
            if cn > 0 then s = dot / (q_norm * cn) end
        end
        scores[label] = s
        if s > best_score then
            best_score = s
            best_label = label
        end
    end

    return best_label, best_score, scores
end

--- @return integer  Number of labels the router was fit on.
function Router:n_labels() return #self._labels end

--- @return integer  Vocab size — distinct terms observed at fit time.
function Router:vocab_size()
    local n = 0
    for _ in pairs(self._idf) do n = n + 1 end
    return n
end

return M
