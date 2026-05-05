# ion7-rag architecture

The technical decisions behind ion7-rag, aimed at contributors and curious
users. The user-facing API surface lives in [`README.md`](README.md) ; the
v1 plan and the literature it is grounded in live in [`RESEARCH.md`](RESEARCH.md).

This document is a **living draft** that grows phase by phase as the v1
plan ships. v0.1.0-alpha1 covers only sections 1, 2 and 8 ; the others
are scaffolded so future PRs can fill them in without restructuring.

---

## 1. Layered overview

Four tiers, top to bottom :

1. **Your Lua application.** Anything from a 30-line CLI prompt-and-answer
   loop to a long-running agentic workflow. ion7-rag is meant to be
   embedded — it exposes data classes and orchestration objects, never
   long-running server processes.

2. **`ion7.rag` — the RAG pipeline.** Pure Lua. Owns five families :
   - **Ingestion** : `loader/*` (text, markdown, html ; v1.1 will add
     eml, json, pdf), `chunk/*` (recursive default ; meta and late
     chunkers as opt-ins).
   - **Storage** : `db/*` over two SQLite files — `chunks.db` for the
     canonical text and metadata ; `index.db` for sqlite-vec dense and
     FTS5 lexical indexes, ATTACHed as `idx`.
   - **Contextual Retrieval** : `context/*` runs a small contextualizer
     model on a dedicated `ion7-llm.Pool`, prepending 50-100 tokens of
     document-level context to each chunk before embedding and FTS
     indexing.
   - **Retrieval and reranking** : `retrieve.lua` glue, `fusion/*`
     (RRF, DBSF, CC), `rerank/*` (cross-encoder default, listwise
     opt-in via ion7-grammar), `route/*` (TF-IDF + classifier baseline).
   - **Generation-time** : `agent/*` (CRAG default, Self-RAG opt-in),
     `pipeline.lua` glue, `eval/*` (RAGAs metrics + Lynx hallucination).

3. **`ion7.core`, `ion7.llm`, `ion7.grammar` — the substrate.**
   ion7-rag is a strict consumer. Embeddings and reranker scoring go
   through ion7-core ; the contextualizer and answer generation go
   through ion7-llm ; constrained-output paths (Self-RAG reflection,
   listwise reranker permutations, future KG-extraction) go through
   ion7-grammar's GBNF.

4. **SQLite + sqlite-vec + FTS5.** Single-machine substrate. brute-force
   on binary-quantized vectors handles up to ~10M chunks comfortably,
   well above the local-first niche ion7-rag targets.

---

## 2. Module organisation

```
src/ion7/rag/
├── init.lua            -- lazy façade : class registry + sub-namespaces
│
├── loader/             -- ingestion : input format → normalized Doc
│   ├── init.lua        -- common loader interface
│   ├── text.lua        -- passthrough
│   ├── markdown.lua    -- via cmark-lua
│   └── html.lua        -- via lua-gumbo
│   -- v1.1 : eml, json, pdf
│
├── chunk/              -- chunkers : Doc → chunks
│   ├── init.lua
│   ├── recursive.lua   -- default, LPeg-based 512-tok / 15-25% overlap
│   ├── meta.lua        -- Meta-Chunking (perplexity), opt-in
│   └── late.lua        -- late chunking, opt-in (needs per-token bridge)
│
├── context/            -- Anthropic Contextual Retrieval
│   ├── init.lua        -- ContextualEnricher
│   └── prompts.lua     -- contextualizer prompt templates
│
├── db/                 -- two-DB SQLite store
│   ├── init.lua        -- open + ATTACH idx, schema migrations
│   ├── schema.lua      -- versioned migrations
│   ├── chunks.lua      -- chunks.db tables (chunks, docs, citations)
│   ├── vec.lua         -- idx.chunks_vec (sqlite-vec : binary + fp32)
│   └── lex.lua         -- idx.chunks_fts (FTS5 contentless)
│
├── fusion/             -- hybrid retrieval fusion
│   ├── init.lua
│   ├── rrf.lua         -- default
│   ├── dbsf.lua        -- distribution-based score fusion
│   └── cc.lua          -- convex combination
│
├── retrieve.lua        -- glue : route → embed query → vec+lex → fuse
│
├── rerank/
│   ├── init.lua
│   ├── crossenc.lua    -- Qwen3-Reranker-0.6B via ion7-core
│   └── listwise.lua    -- LLM listwise via ion7-grammar (opt-in)
│
├── route/              -- adaptive query routing
│   ├── init.lua
│   ├── tfidf.lua       -- TF-IDF + tiny classifier (RAGRouter-Bench baseline)
│   └── llm.lua         -- LLM zero-shot router via grammar (alt)
│
├── hype.lua            -- HyPE ingestion : hypothetical questions per chunk
│
├── agent/              -- generation-time RAG loops
│   ├── init.lua
│   ├── crag.lua        -- Corrective RAG, default
│   └── self_rag.lua    -- reflection-token Self-RAG, opt-in
│
├── eval/               -- RAGAs metrics + hallucination scoring
│   ├── init.lua
│   ├── faithfulness.lua
│   ├── relevancy.lua
│   ├── context_precision.lua
│   ├── context_recall.lua
│   └── lynx.lua        -- Lynx-8B hallucination check
│
├── citation.lua        -- chunk → (doc_id, char_start, char_end)
├── pipeline.lua        -- Pipeline.new / :ingest / :ask / :stream
└── util/
    ├── tokenize.lua    -- token counts via ion7-core Vocab
    └── log.lua         -- mirror of ion7.core.util.log
```

Two patterns recur, mirroring the rest of the ion7 stack :

**Lazy façade.** `init.lua` exposes top-level classes and sub-namespaces
through `__index` metatable hooks. First access triggers the require ;
subsequent reads are direct table reads. A consumer that only loads the
chunker never pays the cost of pulling sqlite-vec or the agent loop.

**Pure-data classes.** `Doc` (loader output), `Chunk`, `Hit`, `Response`
are intentionally inert : they store fields, expose helpers, and hand
themselves to orchestration objects for processing. The split keeps
the unit-test surface large and the integration-test surface bounded.

---

## 3. The two-DB schema

> *Section to be filled at phase 1.*

The truth tier (`chunks.db`) and the index tier (`index.db`) are kept in
separate files so the index can be nuked and rebuilt from scratch
without touching the canonical chunk text or its provenance ledger.
SQLite's `ATTACH DATABASE` makes them appear as one logical schema
(`main` and `idx`) on the same connection.

---

## 4. The retrieval hot path

> *Section to be filled at phases 3-4.*

---

## 5. Contextual Retrieval

> *Section to be filled at phase 5.*

Sketch : the contextualizer model lives behind a dedicated
`ion7-llm.Pool` ; for a document with N chunks, the document's full
text is shared as a prefix across all N contextualization calls and
ion7-llm's RadixAttention exact-match cache amortises the prefill cost
to roughly 1× decode for the whole document.

---

## 6. Generation-time agentic loops

> *Section to be filled at phase 8.*

---

## 7. Cooperation with the ion7 stack

ion7-rag is a **consumer** of ion7-core, ion7-llm and ion7-grammar — not
a fork, not a re-implementation. Concrete rules :

1. **No re-implementation.** Anything the lower layers already expose
   (embedder, reranker scoring, samplers, KV cache primitives, chat
   templates, JSON-Schema → GBNF, threadpool, RadixAttention) is reached
   through the public surface. ion7-rag does not poke at private fields.
2. **No external Lua dependencies for things ion7-core covers.** JSON,
   UTF-8, base64, log routing — all routed through `ion7.vendor.*` or
   `ion7.core.util.*`. Downstream consumers install the four ion7
   modules + lsqlite3 + lpeg ; nothing else.
3. **Contracts at the boundary.** When a feature requires a specific
   ion7-core or ion7-llm capability (e.g. per-token embedding output for
   late chunking), the dependency is documented at the
   call site and the upstream addition lands in that repository before
   ion7-rag ships the consuming code.

---

## 8. Versioning and stability contract

`ion7-rag` follows semver at the level of the public Lua API
(`ion7.rag` and its subordinate modules). Until the first non-alpha
release, the API surface is explicitly unstable.

The `ion7-rag` rockspec pins minimum versions of `ion7-core`,
`ion7-llm` and `ion7-grammar` ; mismatched versions will refuse to
install rather than producing surprising runtime behaviour.
