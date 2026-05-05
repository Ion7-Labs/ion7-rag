--- @module ion7.rag.pipeline
--- @author  ion7 / Ion7 Project Contributors
---
--- High-level orchestrator that ties every ion7-rag layer together.
--- The caller constructs the parts they need (a `db.Handle`, an
--- embedder, optionally an enricher / hype generator / reranker /
--- answerer) and hands them to `Pipeline.new` ; the pipeline owns the
--- wiring, never the lifecycle of those parts.
---
--- Three operations on a populated pipeline :
---
---   pipe:ingest(docs_or_paths, opts?)    load → chunk → enrich → embed → index
---   pipe:retrieve(query, opts?)          embed query → vec + lex → fuse → rerank → hydrate
---   pipe:ask(query, opts?)               retrieve → augment prompt → generate
---
--- The Pipeline takes no model handle. Each ion7-rag and ion7-llm
--- class that does (Engine, Embed, Enricher, ...) is constructed by
--- the caller and passed in. This keeps the constructor small and the
--- ownership rules unambiguous.

local loader      = require "ion7.rag.loader"
local recursive   = require "ion7.rag.chunk.recursive"
local chunks_db   = require "ion7.rag.db.chunks"
local lex_db      = require "ion7.rag.db.lex"
local hype_db     = require "ion7.rag.db.hype"
local rag_embed   = require "ion7.rag.embed"
local retrieve    = require "ion7.rag.retrieve"

local M = {}

-- ── Class ───────────────────────────────────────────────────────────────

local Pipeline = {}
Pipeline.__index = Pipeline
M.Pipeline = Pipeline

-- ── Defaults ────────────────────────────────────────────────────────────

Pipeline.DEFAULTS = {
    chunker = {
        target_tokens  = 512,
        overlap_tokens = 64,
        min_tokens     = 150,
    },
    retrieve = {
        k_dense = 50,
        k_lex   = 50,
        k_final = 10,
        fusion  = "rrf",
        -- 4:1 dense-to-lex prior from Anthropic's Contextual Retrieval
        -- experiments — a sensible starting point for English / French
        -- corpora ; tune `retrieve_opts.weights` if it doesn't fit.
        weights = { dense = 4, lex = 1 },
    },
    -- When a reranker is configured, retrieval pulls this many candidates
    -- and the reranker picks the final `k_final` from that set.
    rerank_top_k  = 20,
    -- Number of excerpts handed to the answerer in `:ask`.
    augment_top_k = 6,
}

local DEFAULT_ANSWER_SYSTEM = [[You are a helpful assistant. Answer the user's question using ONLY the provided excerpts. If the answer is not contained in the excerpts, say so. Cite each excerpt by its number when you use it.]]

local DEFAULT_AUGMENT_TEMPLATE = [[The following excerpts were retrieved from a corpus :

%s

Question : %s

Answer using only the excerpts above.]]

-- ── Constructor ─────────────────────────────────────────────────────────

--- Build a Pipeline.
---
--- Only `handle` and `embedder` are required ; every other slot is
--- optional and gates a specific feature off when omitted (no
--- enrichment, no HyPE, no rerank step, no `:ask` generation).
---
--- @param  opts table {
---     handle           ion7.rag.db.Handle              REQUIRED. Open
---                       chunks.db / index.db pair (see `ion7.rag.db.open`).
---     embedder         ion7.llm.Embed                  REQUIRED. Used to
---                       embed chunk text at ingest and queries at retrieve.
---     enricher         ion7.rag.context.Enricher?      When set, every
---                       ingested chunk gets a contextual prefix written
---                       to `chunks.contextual_text`.
---     hype_generator   ion7.rag.hype.Generator?        When set, every
---                       ingested chunk spawns N hypothetical-question
---                       embeddings stored in `idx.hype_vec`.
---     hype_n           integer?                        Number of HyPE
---                       questions per chunk. Default 3.
---     reranker         { :rerank(handle, q, hits, k) }? Cross-encoder
---                       reranker called between retrieval and final
---                       truncation.
---     answerer         { engine, cm }?                 ion7-llm engine
---                       pair used by `:ask` to generate the final
---                       response.
---     chunker_opts     table?  Override `Pipeline.DEFAULTS.chunker`
---                       (target_tokens, overlap_tokens, min_tokens).
---     retrieve_opts    table?  Override `Pipeline.DEFAULTS.retrieve`
---                       (k_dense, k_lex, k_final, fusion, weights,
---                       vec_tier, fusion_k).
---     rerank_top_k     integer? Override `Pipeline.DEFAULTS.rerank_top_k`.
---     augment_top_k    integer? Override `Pipeline.DEFAULTS.augment_top_k`.
---     count_tokens     function(string) → integer?     Token counter
---                       used by the chunker. Defaults to a
---                       4-chars-per-token approximation ; pass the
---                       embedder vocab's tokenizer for an exact count.
---     answer_system    string?  Override the system prompt used in `:ask`.
---     augment_template string?  Override the RAG augmentation template
---                       (must carry two `%s` for excerpts + question).
--- }
--- @return ion7.rag.Pipeline
function Pipeline.new(opts)
    opts = opts or {}
    local handle   = assert(opts.handle,   "Pipeline.new : opts.handle required")
    local embedder = assert(opts.embedder, "Pipeline.new : opts.embedder required")

    local chunker_opts  = opts.chunker_opts  or {}
    local retrieve_opts = opts.retrieve_opts or {}

    -- Fold defaults into the opts tables we cache.
    local chunker_cfg = {
        target_tokens  = chunker_opts.target_tokens  or Pipeline.DEFAULTS.chunker.target_tokens,
        overlap_tokens = chunker_opts.overlap_tokens or Pipeline.DEFAULTS.chunker.overlap_tokens,
        min_tokens     = chunker_opts.min_tokens     or Pipeline.DEFAULTS.chunker.min_tokens,
        count_tokens   = opts.count_tokens or function(s) return math.ceil(#s / 4) end,
    }
    local retrieve_cfg = {
        k_dense = retrieve_opts.k_dense or Pipeline.DEFAULTS.retrieve.k_dense,
        k_lex   = retrieve_opts.k_lex   or Pipeline.DEFAULTS.retrieve.k_lex,
        k_final = retrieve_opts.k_final or Pipeline.DEFAULTS.retrieve.k_final,
        fusion  = retrieve_opts.fusion  or Pipeline.DEFAULTS.retrieve.fusion,
        weights = retrieve_opts.weights or Pipeline.DEFAULTS.retrieve.weights,
        vec_tier = retrieve_opts.vec_tier,
        fusion_k = retrieve_opts.fusion_k,
    }

    return setmetatable({
        _handle         = handle,
        _embedder       = embedder,
        _enricher       = opts.enricher,
        _hype_generator = opts.hype_generator,
        _hype_n         = opts.hype_n or 3,
        _reranker       = opts.reranker,
        _answerer       = opts.answerer,
        _chunker        = chunker_cfg,
        _retrieve       = retrieve_cfg,
        _rerank_top_k   = opts.rerank_top_k  or Pipeline.DEFAULTS.rerank_top_k,
        _augment_top_k  = opts.augment_top_k or Pipeline.DEFAULTS.augment_top_k,
        _answer_system    = opts.answer_system    or DEFAULT_ANSWER_SYSTEM,
        _augment_template = opts.augment_template or DEFAULT_AUGMENT_TEMPLATE,
    }, Pipeline)
end

--- Read the next available `hype_id` from `idx.hype_vec`. One COUNT
--- query per ingest run ; the result is incremented locally as new
--- HyPE rows are produced.
local function _next_hype_id(handle)
    local stmt = handle:prepare("SELECT COALESCE(MAX(hype_id), 0) FROM idx.hype_vec")
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n + 1
end

-- ── Internal : ensure we have a Doc to work with ───────────────────────

--- Coerce an ingest input to a `Doc`. A path string is loaded via
--- `ion7.rag.loader.from_file` ; a table that already carries `text`
--- and `format` is returned as-is.
local function _to_doc(input)
    if type(input) == "string" then
        return loader.from_file(input)
    end
    if type(input) == "table" and input.text and input.format then
        return input
    end
    error("Pipeline : ingest input must be a path string or a Doc table " ..
          "(got " .. type(input) .. ")", 3)
end

-- ── Ingest ─────────────────────────────────────────────────────────────

--- Ingest one or more documents end-to-end : load, chunk, optionally
--- enrich, embed, and index. Each chunk is written to `chunks.db`,
--- embedded into `idx.chunks_vec`, indexed into `idx.chunks_fts`, and
--- (when a HyPE generator is configured) given N hypothetical-question
--- vectors in `idx.hype_vec`.
---
--- @param  inputs  string | Doc | array  A single path, a single Doc
---                  (table with `.text` and `.format`), or a list of either.
--- @param  opts    table? {
---     on_doc     function(doc, idx)?    Called before each doc is processed.
---     on_chunk   function(done, total)? Called after each doc's chunks index.
---     enrich     bool?                  Default `true` when an enricher is
---                  configured, `false` otherwise. Forcing `true` without an
---                  enricher raises.
--- }
--- @return table {
---     docs_ingested  integer  Number of docs processed.
---     chunks_indexed integer  Number of chunk rows written.
---     hype_indexed   integer  Number of HyPE-question rows written.
--- }
--- @raise When `opts.enrich = true` but no enricher was configured at
---        construction.
function Pipeline:ingest(inputs, opts)
    opts = opts or {}
    local list = inputs
    if type(inputs) == "string" or
       (type(inputs) == "table" and inputs.text and inputs.format) then
        list = { inputs }
    end

    local enrich_default = self._enricher ~= nil
    local do_enrich = opts.enrich
    if do_enrich == nil then do_enrich = enrich_default end
    if do_enrich and not self._enricher then
        error("Pipeline:ingest : opts.enrich = true but no enricher was " ..
              "configured at Pipeline.new time", 2)
    end

    local total_docs       = 0
    local total_chunks     = 0
    local total_hype       = 0
    local hype_id_seed     = self._hype_generator and _next_hype_id(self._handle) or 1

    for i, input in ipairs(list) do
        local doc = _to_doc(input)
        if opts.on_doc then opts.on_doc(doc, i) end

        local doc_pk = chunks_db.insert_doc(self._handle, {
            doc_id     = doc.id,
            format     = doc.format,
            source_uri = doc.source_uri,
            title      = doc.title,
        })

        local cs = recursive.chunk(doc, self._chunker)

        if do_enrich then
            self._enricher:enrich_chunks(doc.text, cs)
        end

        -- Insert chunks into the truth tier.
        local chunk_rows = {}
        for j, c in ipairs(cs) do
            chunk_rows[j] = {
                doc_pk          = doc_pk,
                section         = c.section,
                char_start      = c.char_start,
                char_end        = c.char_end,
                n_tokens        = c.n_tokens,
                raw_text        = c.raw_text,
                contextual_text = c.contextual_text,
            }
        end
        local pks = chunks_db.insert_chunks(self._handle, chunk_rows)

        -- Pick the indexed view : contextual_text when present, raw_text
        -- otherwise. Same view feeds both vec and lex so retrieval
        -- weights stay consistent between tiers.
        local index_rows = {}
        local lex_rows   = {}
        for j, c in ipairs(cs) do
            local view = c.contextual_text or c.raw_text
            index_rows[j] = { chunk_id = pks[j], text = view }
            lex_rows[j]   = { chunk_id = pks[j], text = view }
        end
        rag_embed.index_chunks(self._handle, self._embedder, index_rows)
        lex_db.upsert_many(self._handle, lex_rows)

        -- HyPE pass : per-chunk question generation, embedding, indexing.
        if self._hype_generator then
            local hype_rows = {}
            for j, c in ipairs(cs) do
                local questions = self._hype_generator:generate(
                    c.raw_text, self._hype_n)
                if #questions > 0 then
                    local q_vecs = self._embedder:encode_many(questions)
                    for q_idx, q in ipairs(questions) do
                        hype_rows[#hype_rows + 1] = {
                            hype_id   = hype_id_seed,
                            chunk_id  = pks[j],
                            question  = q,
                            embedding = q_vecs[q_idx],
                        }
                        hype_id_seed = hype_id_seed + 1
                    end
                end
            end
            if #hype_rows > 0 then
                hype_db.upsert_many(self._handle, hype_rows)
                total_hype = total_hype + #hype_rows
            end
        end

        if opts.on_chunk then opts.on_chunk(#cs, #cs) end

        total_docs   = total_docs   + 1
        total_chunks = total_chunks + #cs
    end

    return {
        docs_ingested  = total_docs,
        chunks_indexed = total_chunks,
        hype_indexed   = total_hype,
    }
end

-- ── Retrieve ────────────────────────────────────────────────────────────

--- Run a hybrid retrieval and, when a reranker is configured, rerank
--- the candidate set before truncating to `k_final`. Hits are hydrated
--- with `raw_text`, `section`, `doc_pk`, `doc_id`, and `doc_title`
--- from `chunks.db` before being returned, so the caller never has to
--- second-trip the database.
---
--- @param  query  string  Natural-language query.
--- @param  opts   table? {
---     mode        string?  "hybrid" (default) | "dense_only" | "lex_only".
---                  - "hybrid"     : encode query as vector AND pass it to FTS5.
---                  - "dense_only" : encode + vector search only.
---                  - "lex_only"   : FTS5 only ; skips the embedder forward
---                                   pass on the query (cheaper hot path).
---     query_text  string?  FTS5 query string in "hybrid" / "lex_only"
---                  modes. Defaults to `query` ; override when the
---                  user-facing question and the FTS5-friendly form
---                  differ (escaping, expansion, ...).
---     k_dense     integer?  Override `retrieve_opts.k_dense` for this call.
---     k_lex       integer?
---     k_hype      integer?
---     k_final     integer?
---     fusion      string?   Override the fusion strategy.
---     fusion_k    number?   RRF constant override.
---     weights     table?    Per-source fusion weights.
---     vec_tier    string?   "full" | "binary".
---     use_hype    bool|"auto"?
--- }
--- @return Hit[]  Hydrated hits ordered by descending score :
---                `{ chunk_id, score, prior_score?, raw_text, section,
---                  doc_pk, doc_id, doc_title }`.
--- @raise When `opts.mode` is not one of the three accepted values.
function Pipeline:retrieve(query, opts)
    opts = opts or {}

    local mode = opts.mode or "hybrid"
    if mode ~= "hybrid" and mode ~= "dense_only" and mode ~= "lex_only" then
        error("Pipeline:retrieve : opts.mode must be 'hybrid' | 'dense_only' | 'lex_only', got '"
              .. tostring(mode) .. "'", 2)
    end

    local cfg = {}
    for k, v in pairs(self._retrieve) do cfg[k] = v end
    for k, v in pairs(opts)            do cfg[k] = v end

    -- When a reranker is configured, pull a wider candidate set and
    -- let the reranker pick the final top-K from there.
    local fetch_k = cfg.k_final
    if self._reranker then
        fetch_k = math.max(self._rerank_top_k, cfg.k_final)
    end

    local search_opts = {
        k_dense    = cfg.k_dense,
        k_lex      = cfg.k_lex,
        k_hype     = cfg.k_hype,
        k_final    = fetch_k,
        fusion     = cfg.fusion,
        fusion_k   = cfg.fusion_k,
        weights    = cfg.weights,
        vec_tier   = cfg.vec_tier,
        use_hype   = cfg.use_hype,
    }

    if mode == "hybrid" or mode == "dense_only" then
        search_opts.query_vec = self._embedder:encode(query)
    end
    if mode == "hybrid" or mode == "lex_only" then
        search_opts.query_text = opts.query_text or query
    end

    local hits = retrieve.search(self._handle, search_opts)

    if self._reranker then
        hits = self._reranker:rerank(self._handle, query, hits, cfg.k_final)
    end

    -- Hydrate raw_text, section, and doc identity from the truth tier
    -- so the caller has every field needed to cite or display the hit.
    local out = {}
    for i, hit in ipairs(hits) do
        local row = chunks_db.get_chunk(self._handle, hit.chunk_id)
        if row then
            local doc_row = chunks_db.get_doc_by_pk(self._handle, row.doc_pk)
            out[i] = {
                chunk_id    = hit.chunk_id,
                score       = hit.score,
                prior_score = hit.prior_score,
                raw_text    = row.raw_text,
                section     = row.section,
                doc_pk      = row.doc_pk,
                doc_id      = doc_row and doc_row.doc_id or nil,
                doc_title   = doc_row and doc_row.title  or nil,
            }
        end
    end

    return out
end

-- ── Ask ────────────────────────────────────────────────────────────────

--- Format the top-`top_k` hits as a numbered list of cited excerpts
--- ready to be plugged into the augment template.
local function _format_excerpts(hits, top_k)
    local n = math.min(#hits, top_k)
    local parts = {}
    for i = 1, n do
        local h = hits[i]
        local cite = h.doc_id or ("chunk-" .. tostring(h.chunk_id))
        if h.section and h.section ~= "" then
            cite = cite .. " > " .. h.section
        end
        parts[i] = string.format("[%d] (%s)\n%s", i, cite, h.raw_text)
    end
    return table.concat(parts, "\n\n")
end

--- Run a Retrieval-Augmented Generation cycle : retrieve hits for the
--- query, format them as excerpts, prompt the answerer, and return the
--- generated response together with the supporting hits.
---
--- @param  query  string
--- @param  opts   table? {
---     retrieve   table?    Pass-through overrides for the inner
---                  `:retrieve` call (k_final, fusion, weights, ...).
---     max_tokens integer?  Generation cap. Default 512.
---     sampler    sampler?  Override the answerer's default sampler.
---     top_k      integer?  Number of excerpts handed to the model.
---                  Defaults to the Pipeline's `augment_top_k`.
--- }
--- @return Response  Result of `ion7.llm.Engine:chat` (carries `content`,
---                   `tool_calls`, and the rest of the Response shape).
--- @return Hit[]     The retrieved hits used to build the prompt.
--- @raise When no answerer was configured at construction.
function Pipeline:ask(query, opts)
    if not self._answerer then
        error("Pipeline:ask : no answerer was configured at Pipeline.new time " ..
              "(opts.answerer = { engine = ..., cm = ... })", 2)
    end
    opts = opts or {}

    local hits = self:retrieve(query, opts.retrieve)
    local top_k = opts.top_k or self._augment_top_k
    local prompt = string.format(self._augment_template,
        _format_excerpts(hits, top_k), query)

    local llm = require "ion7.llm"
    local engine = self._answerer.engine
    local cm     = self._answerer.cm
    cm:set_system(self._answer_system)

    local session = llm.Session.new()
    session:add_user(prompt)

    local response = engine:chat(session, {
        max_tokens = opts.max_tokens or 512,
        sampler    = opts.sampler,
    })

    cm:release(session)

    return response, hits
end

-- ── Accessors ──────────────────────────────────────────────────────────

--- Underlying database handle.
--- @return ion7.rag.db.Handle
function Pipeline:handle()   return self._handle   end

--- Embedder bound to this pipeline.
--- @return ion7.llm.Embed
function Pipeline:embedder() return self._embedder end

return M
