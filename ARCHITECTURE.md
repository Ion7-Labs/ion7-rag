# ion7-rag architecture

The technical decisions behind ion7-rag, aimed at contributors and curious
users. The user-facing API surface lives in [`README.md`](README.md).

---

## 1. Layered overview

Four tiers, top to bottom :

1. **Your Lua application.** Anything from a 30-line CLI prompt-and-answer
   loop to a long-running agentic workflow. ion7-rag is meant to be
   embedded — it exposes data classes and orchestration objects, never
   long-running server processes.

2. **`ion7.rag` — the RAG pipeline.** Pure Lua. Owns five families :
   - **Ingestion** : `loader/*` (text, markdown via cmark, html via
     gumbo), `chunk/*` (recursive default and late chunker).
   - **Storage** : `db/*` over two SQLite files — `chunks.db` for the
     canonical text and metadata ; `index.db` for sqlite-vec dense, FTS5
     lexical, and HyPE vec0 indexes, ATTACHed as `idx`.
   - **Contextual Retrieval** : `context/*` runs a small contextualizer
     model on top of `ion7-llm.Engine`, prepending 50–100 tokens of
     document-level context to each chunk before embedding and FTS
     indexing.
   - **Retrieval and reranking** : `retrieve.lua` glue, `fusion/*` (RRF,
     DBSF, CC), `rerank/*` (pointwise yes/no logprob judge), `route/*`
     (TF-IDF + per-class centroid classifier).
   - **Generation-time** : `agent/*` (CRAG and Self-RAG), `pipeline.lua`
     orchestrator, `eval/*` (Faithfulness, ContextPrecision, Lynx).

3. **`ion7.core`, `ion7.llm`, `ion7.grammar` — the substrate.**
   ion7-rag is a strict consumer. Embeddings and reranker scoring go
   through ion7-core ; the contextualizer and answer generation go
   through ion7-llm ; constrained-output paths (Self-RAG reflection
   tokens) go through ion7-grammar's GBNF.

4. **SQLite + sqlite-vec + FTS5.** Single-machine substrate. Brute-force
   on binary-quantized vectors handles up to ~10M chunks comfortably,
   well above the local-first niche ion7-rag targets.

---

## 2. Module organisation

```
src/ion7/rag/
├── init.lua            -- lazy façade : class registry + sub-namespaces
│
├── loader/             -- ingestion : input format → normalized Doc
│   ├── init.lua        -- format detection + loader registry
│   ├── text.lua        -- passthrough
│   ├── markdown.lua    -- via cmark (sections from H1-H6 hierarchy)
│   └── html.lua        -- via gumbo (sections from heading hierarchy)
│
├── chunk/              -- chunkers : Doc → Chunk[]
│   ├── init.lua        -- chunker registry
│   ├── recursive.lua   -- separator-hierarchy splitter (default)
│   └── late.lua        -- Günther late chunking via per-token embedder
│
├── context/            -- Anthropic Contextual Retrieval
│   ├── init.lua        -- Enricher class
│   └── prompts.lua     -- contextualizer prompt templates
│
├── db/                 -- two-DB SQLite store
│   ├── init.lua        -- open + ATTACH idx, sqlite-vec extension load
│   ├── schema.lua      -- versioned migrations + meta table
│   ├── chunks.lua      -- main.docs / main.chunks (canonical truth)
│   ├── vec.lua         -- idx.chunks_vec (sqlite-vec : binary + fp32)
│   ├── lex.lua         -- idx.chunks_fts (FTS5 contentless BM25)
│   └── hype.lua        -- idx.hype_vec (HyPE Option B aux columns)
│
├── embed.lua           -- query / chunk embedding via ion7-core
├── hype.lua            -- HyPE Generator : per-chunk hypothetical
│                          questions for retrieval-time alignment
│
├── retrieve.lua        -- glue : embed query → vec + lex + hype → fuse
│
├── fusion/             -- hybrid retrieval fusion
│   ├── init.lua        -- strategy registry
│   ├── rrf.lua         -- reciprocal rank fusion
│   ├── dbsf.lua        -- distribution-based score fusion
│   └── cc.lua          -- min-max convex combination
│
├── rerank/
│   ├── init.lua        -- reranker registry
│   └── pointwise.lua   -- LLM yes/no logprob judge (Qwen3-Reranker style)
│
├── route/              -- adaptive query routing
│   ├── init.lua        -- router registry
│   └── tfidf.lua       -- TF-IDF + per-class centroid classifier
│
├── agent/              -- generation-time RAG control loops
│   ├── init.lua        -- agent registry
│   ├── prompts.lua     -- shared reformulation / reflection prompts
│   ├── crag.lua        -- Corrective RAG (Yan et al., 2024)
│   └── self_rag.lua    -- reflection-token Self-RAG via ion7-grammar
│
├── eval/               -- reference-free RAGAs metrics
│   ├── init.lua        -- metric registry
│   ├── prompts.lua     -- judge prompt templates
│   ├── faithfulness.lua    -- claims-grounded-in-contexts ratio
│   ├── context_precision.lua -- rank-weighted relevance precision
│   └── lynx.lua        -- Patronus Lynx PASS / FAIL judge
│
├── pipeline.lua        -- Pipeline.new / :ingest / :retrieve / :ask
└── util/
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

The truth tier (`chunks.db`) and the index tier (`index.db`) are kept in
separate files so the index can be nuked and rebuilt from scratch
without touching the canonical chunk text or its provenance ledger.
SQLite's `ATTACH DATABASE` makes them appear as one logical schema
(`main` and `idx`) on the same connection.

**`chunks.db` (main).**

- `meta(key TEXT PK, value TEXT)` — schema version, ingest config.
- `docs(id, source_uri, format, title, meta_json, ingested_at)` —
  one row per ingested document.
- `chunks(id, doc_id, ord, section, char_start, char_end, raw_text,
  contextual_text, n_tokens)` — one row per chunk, ordered within doc.
  `contextual_text` is non-null only when Contextual Retrieval is
  enabled at ingest time.

**`index.db` (idx).**

- `idx.chunks_vec` — sqlite-vec virtual table with two embedding
  columns : a 192-d binary shortlist for fast brute-force candidate
  selection and a 1024-d fp32 column for the rerank tier.
- `idx.chunks_fts` — FTS5 contentless table indexing
  `coalesce(contextual_text, raw_text)` so BM25 sees the enriched
  view when it exists.
- `idx.hype_vec` — sqlite-vec aux table for HyPE Option B : one row per
  hypothetical question, with chunk_id and question_index aux columns.

The `Handle` returned by `db.open` exposes connection-level helpers
(`exec`, `prepare`, `transaction`, `close`) and forwards strict typing
through CAST clauses where vec0 demands it.

---

## 4. The retrieval hot path

`Pipeline:retrieve(query, opts)` runs in five steps :

1. **Embed the query** through `embed.lua`. The embedder Context is
   pinned for the life of the Pipeline.
2. **Dense search** via `db.vec.knn_binary` — Hamming-distance brute force
   over the 192-d binary column, returning the top-K candidate
   `chunk_id`s.
3. **Lex search** via `db.lex` — FTS5 BM25 over the same query string,
   trigram tokenizer optional via `db.lex` opts.
4. **HyPE search** (optional) via `db.hype.knn` — same query embedding
   matched against the hypothetical-question vectors, surfaced as a
   third source.
5. **Fuse** via `fusion/*` — RRF (default), DBSF, or CC. Weights are
   passed in opts ; the 4:1 dense:lex prior is the documented default.

`Pipeline:ask(query)` extends this with a reranker pass (`rerank.Pointwise`
when configured) and an answerer pass (`ion7-llm.Engine` with the
augmentation template).

---

## 5. Contextual Retrieval

The contextualizer model lives behind a dedicated `ion7-llm.Engine`. For
a document with N chunks, the document's full text is shared as a prefix
across all N contextualization calls and ion7-llm's RadixAttention
exact-match cache amortises the prefill cost.

`Enricher:enrich_chunks(full_doc_text, chunks)` mutates each chunk in
place by setting `chunk.contextual_text = "<context>\n\n<raw_text>"`,
which downstream embedders and FTS pick up via
`coalesce(contextual_text, raw_text)`.

---

## 6. Generation-time agentic loops

Both agents wrap an existing `Pipeline` and never mutate it.

**CRAG** (`agent.crag`) — corrective. Retrieve → rerank-score the top-K
candidates → bin into high / low / mixed confidence by score
thresholds → on low confidence, reformulate the query through the
answerer engine and retry up to `max_retries` times → answer through
the Pipeline's existing answer template, with a confidence caveat
appended on mixed / exhausted-low branches.

**Self-RAG** (`agent.self_rag`) — reflective. Three reflection points,
each returning a JSON object grammar-constrained via `ion7.grammar` :

1. `retrieve_decision` — should this query hit the corpus at all ?
2. `relevance_grade` — per-hit yes/no relevance check ; irrelevant
   hits are dropped before generation, with a fallback to all hits if
   too few survive.
3. `support_grade` — post-answer audit : full / partial / none.

Self-RAG's grammar uses the canonical DCCD-style sampler order
(grammar → top_k → dist) so the grammar masks the full vocab before
any candidate-pruning step shrinks the candidate set under it.

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
   modules + `lsqlite3` + `lpeg` ; nothing else.
3. **Contracts at the boundary.** When a feature requires a specific
   ion7-core or ion7-llm capability (e.g. per-token embedding output for
   late chunking via `Context:decode_for_embeddings` and
   `Context:embedding_token_ptr`), the dependency is documented at
   the call site.

---

## 8. Versioning and stability contract

`ion7-rag` follows semver at the level of the public Lua API
(`ion7.rag` and its subordinate modules). Until the first non-alpha
release, the API surface is explicitly unstable.

The `ion7-rag` rockspec pins minimum versions of `ion7-core`,
`ion7-llm` and `ion7-grammar` ; mismatched versions will refuse to
install rather than producing surprising runtime behaviour.
