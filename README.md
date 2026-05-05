<div align="center">

# ion7-rag

**Local-first Retrieval-Augmented Generation as a pure-Lua library, on top of [ion7-core](https://github.com/Ion7-Labs/ion7-core), [ion7-llm](https://github.com/Ion7-Labs/ion7-llm) and [ion7-grammar](https://github.com/Ion7-Labs/ion7-grammar).**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![LuaJIT 2.1](https://img.shields.io/badge/LuaJIT-2.1-orange.svg)](https://luajit.org/)
[![Status: alpha](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

</div>

---

`ion7-core` gives you embeddings, rerankers, generation. `ion7-llm` gives you
the chat pipeline, multi-session pool, prefix cache. `ion7-grammar` gives you
constrained output. `ion7-rag` is the layer that wires those into a working
RAG system : ingestion, chunking, contextual retrieval, hybrid search, fusion,
reranking, agentic loops, evaluation.

## Status

**v0.1.0-alpha1 — bootstrap.** The repo skeleton, test harness, façade and
build system are in place. Real functionality lands phase by phase against
the plan tracked in [`RESEARCH.md`](RESEARCH.md). Today's surface is
intentionally minimal — only the lazy façade and the logging helper.

## Design at a glance

- **Storage substrate.** Two SQLite files. `chunks.db` holds the canonical
  text, document metadata, provenance and citations — the source of truth.
  `index.db` holds a `sqlite-vec` dense vector store (binary 192-d shortlist
  + fp32 1024-d rerank tier) and an FTS5 BM25 index — disposable and
  rebuildable from `chunks.db`.
- **Hybrid retrieval.** RRF (default), DBSF, Convex Combination as fusion
  options over (dense + lexical) candidate lists. 4:1 dense:BM25 prior from
  Anthropic's Contextual Retrieval results.
- **Contextual Retrieval.** A small contextualizer model running through a
  dedicated `ion7-llm.Pool` rewrites each chunk with 50-100 tokens of
  document-level context before embedding and FTS indexing. ion7-llm's
  RadixAttention exact-match prefix cache amortises the per-document cost
  almost completely.
- **Constrained-output advantage.** Self-RAG reflection tokens, listwise
  reranker permutations, KG-extraction triples for future graph-RAG — all
  go through `ion7-grammar`'s GBNF, so local 7-8B models emit valid output
  by construction, not by retry.
- **Library, not a server.** No HTTP, no SSE, no CLI binary, no daemon.
  ion7-rag is meant to be embedded.

## Quick taste

The high-level `Pipeline` API lands at phase 6 of the v1 plan ; the snippet
below is a forward-looking sketch.

```lua
local ion7 = require "ion7.core"
local llm  = require "ion7.llm"
local rag  = require "ion7.rag"

ion7.init({ log_level = 0 })

local model = ion7.Model.load(os.getenv("ION7_EMBED_MODEL"))
local pipe  = rag.Pipeline.new({
    model        = model,
    chunks_db    = "./data/chunks.db",
    index_db     = "./data/index.db",
    contextual   = true,         -- enable Anthropic Contextual Retrieval
    fusion       = "rrf",        -- or "dbsf", "cc"
})

pipe:ingest({ "./docs/handbook.md", "./docs/policies/" })

for chunk in pipe:stream("How do reservations work past the cutoff ?") do
    if     chunk.kind == "content"   then io.write(chunk.text)
    elseif chunk.kind == "citation"  then io.write(string.format(
        "  [%s § %s]\n", chunk.doc_id, chunk.section)) end
end

ion7.shutdown()
```

## Documentation

- [`RESEARCH.md`](RESEARCH.md) — 2024-2026 RAG state-of-the-art survey ;
  the v1 shortlist this repo implements.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — layered design, two-DB schema,
  retrieval flow, runtime cooperation with the rest of the ion7 stack.
- [`INSTALL.md`](INSTALL.md) — install paths, sibling-checkout layouts,
  troubleshooting.

## Compatibility

| Component     | Requirement                                          |
|---------------|------------------------------------------------------|
| LuaJIT        | 2.1 (any post-2017 build)                            |
| ion7-core     | matched release ; embedder + reranker bridge needed  |
| ion7-llm      | matched release                                      |
| ion7-grammar  | matched release ; only required by constrained paths |
| `lsqlite3`    | 0.9.5+ with `sqlite-vec` extension loadable          |
| `lpeg`        | 1.0+                                                 |
| OS            | whatever ion7-core builds on (Linux glibc, macOS 12+) |

## License

[MIT](LICENSE). ion7-rag builds on
[ion7-core](https://github.com/Ion7-Labs/ion7-core),
[ion7-llm](https://github.com/Ion7-Labs/ion7-llm) and
[ion7-grammar](https://github.com/Ion7-Labs/ion7-grammar) — themselves built
on [llama.cpp](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and
contributors.
