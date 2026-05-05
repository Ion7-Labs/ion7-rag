# ion7-rag — RAG state-of-the-art survey

**Snapshot date :** 2026-05-04
**Purpose :** input for the v1 design of `ion7-rag`. Captures the 2024-2026 RAG
literature, ranks techniques by cost/benefit on our stack, and calls out what
has been superseded. This is a frozen reference, not a living spec — re-run the
survey before any v2 cycle.

**Scope filter.** `ion7-rag` is a LuaJIT 2.1 library on top of :

- `ion7-core` — LuaJIT FFI to llama.cpp + a thin C++ bridge (chat templates,
  JSON-Schema → GBNF, samplers, speculative decoding, embeddings).
- `ion7-llm` — pure-Lua chat pipeline. Already ships `Embed` (single-text
  encode + cosine), `Engine`, `Pool`, KV management with prefix cache,
  RadixAttention exact-match cache, mid-gen eviction.
- `ion7-grammar` — pure-Lua grammar engine producing GBNF for constrained
  generation.
- Storage substrate already chosen : `lsqlite3` + `sqlite-vec` + SQLite FTS5.
  HTML5 via `lua-gumbo`, Markdown via `cmark-lua`, sentence/citation parsing
  via LPeg, PDF via FFI to libmupdf or shell to `pdftotext`.
- Local-first / embedded niche. Library, not a server.

The implementability tags below (*easy / moderate / hard / not feasible*) are
relative to that substrate, not to a generic Python-RAG stack.

---

## Table of contents

1. [Chunking strategies](#1-chunking-strategies)
2. [Embedding models](#2-embedding-models)
3. [Retrieval algorithms](#3-retrieval-algorithms)
4. [Reranking and post-retrieval](#4-reranking-and-post-retrieval)
5. [Generation-time RAG](#5-generation-time-rag)
6. [Evaluation](#6-evaluation)
7. [Indexing and store innovations](#7-indexing-and-store-innovations)
8. [What's overrated or didn't pan out](#8-whats-overrated-or-didnt-pan-out)
9. [v1 shortlist (ranked)](#9-v1-shortlist-ranked)
10. [ion7-specific differentiators](#10-ion7-specific-differentiators)
11. [Sources](#11-sources)

---

## 1. Chunking strategies

### 1.1 Anthropic Contextual Retrieval (Sept 2024) and its 2025 formalization

Anthropic's [blog post](https://www.anthropic.com/news/contextual-retrieval)
(Sept 19 2024) prepends a 50-100 token LLM-generated context to each chunk
*before* embedding and BM25 indexing. The chunking stays the same ; only the
text fed to the encoders is augmented. Reported : ~35 % reduction in
retrieval-failure rate, ~49 % when combined with a reranker. Cost is amortized
at ingest time and dominated by document-level prompt caching (the full doc is
cached once per chunking pass).

Academic follow-up : *Reconstructing Context : Evaluating Advanced Chunking
Strategies for RAG* — Pham et al., [arXiv:2504.19754](https://arxiv.org/abs/2504.19754),
ECIR 2025 Workshop on Knowledge-Enhanced IR (Springer LNCS). The paper
formalizes Contextual Retrieval, compares it head-to-head with late chunking,
and confirms Anthropic's 4:1 dense-to-BM25 fusion ratio empirically. Key
nuance from the 2025 follow-up : the contextualizer should be a *small* fast
model (0.6B-1B) rather than the generation model — the cost/quality curve
flattens fast.

**Implementability for ion7-rag : easy.** Pure prompt orchestration over
`ion7-llm.Pool` (fast contextualizer pool + main pool) + sqlite-vec + FTS5.
RadixAttention's exact-match prefix cache aligns naturally with the
"document prefix shared across all its chunks" pattern — we get the prompt
caching savings essentially for free. **Recommended for v1 default.**

### 1.2 Late chunking — Jina (Sept 2024, revised Jul 2025)

*Late chunking : Contextual Chunk Embeddings Using Long-Context Embedding
Models* — Günther et al., [arXiv:2409.04701](https://arxiv.org/abs/2409.04701)
(v3 revised Jul 7 2025). Embed the full document with a long-context encoder,
then mean-pool the per-token embeddings *per chunk* afterwards. Each chunk
vector contains contextual signal from the whole document without an LLM call.
The 2025 v3 revision extended to 8K-32K windows on Jina v3/v4 and reports that
late chunking *complements* (does not replace) Contextual Retrieval.

2025-2026 derivatives :

- **GraLC-RAG** — graph-aware late chunking with UMLS / document-structure
  graph signals (arXiv:2603.22633).
- **ColChunk / Visual Late Chunking** — late chunking on ColPali-style
  multimodal patch embeddings (arXiv:2604.10167, 2026).

**Implementability : moderate.** Needs a long-context embedder GGUF (Jina
v3/v4 GGUFs exist ; Qwen3-Embedding has 32K context) AND a bridge that
exposes the per-token output embedding matrix (`[n_tokens × n_dim]`) instead
of the pooled vector. ion7-core's `Embed` currently returns the pooled vector
only — adding a per-token accessor is a small bridge change. **Worthwhile for
v1 as an opt-in chunker.**

### 1.3 Meta-Chunking — Zhao et al. (ICLR 2025)

*Meta-Chunking : Learning Efficient Text Segmentation via Logical Perception*
— [arXiv:2410.12788](https://arxiv.org/abs/2410.12788), ICLR 2025. Two
LLM-driven adaptive segmentation algorithms — **Perplexity Chunking** and
**Margin Sampling Chunking** — operate at a "meta-chunk" granularity between
sentence and paragraph. Beats similarity-chunking on 2WikiMultihopQA by 1.32
points using only 45.8 % the compute time.

**Implementability : moderate.** Perplexity chunking is a natural fit for
ion7-core because we already read per-token logprobs from llama.cpp. Margin
sampling needs a small classifier head ; can be done with ion7-grammar
constraining a binary decision. **Worth shipping as an opt-in chunker.**

### 1.4 Recursive splitter reality-check (NAACL 2025 ; FloTorch 2026)

A NAACL 2025 Findings paper and the FloTorch 2026 benchmark both report that
vanilla semantic chunking *underperforms* a well-tuned 512-token recursive
splitter (54 % vs 69 % end-to-end accuracy in FloTorch). Semantic chunkers
tend to over-fragment to a ~43-token mean chunk size ; setting
`min_chunk_size=150` recovers most of the gap. Practical 2026 default :
**RecursiveCharacterTextSplitter at 256-512 tokens, 10-25 % overlap**. Azure
recommends 512/25 % ; FloTorch finds factoid queries best at 256-512 and
multi-hop at 512-1024.

**Implementability : easy.** Pure LPeg. **Ship as the default chunker.**

### 1.5 RAPTOR and 2025 extensions

RAPTOR (Sarthi et al., ICLR 2024) — recursive cluster-then-summarize tree.
Still relevant. *Frontiers in Computer Science* (2025) "Enhancing RAPTOR with
semantic chunking and adaptive graph clustering" replaces fixed-token leaves
with semantic segmentation and adds an adaptive Leiden-clustering layer.
RAGFlow integrated long-context-RAG-on-RAPTOR in 2025.

**Implementability : moderate.** Tree construction needs k-means /
agglomerative clustering in Lua (or a small FFI kernel). Summarization fits
ion7-llm. Storage as `(chunk_id, parent_id, level)` rows in SQLite is
trivial. **Defer to v1.5 ; not a v1 must-have.**

### 1.6 Layout-aware parsers — Docling, MinerU, Marker (2024-2026)

Docling (IBM, LF AI&Data Foundation) uses DocLayNet + TableFormer ; MinerU
uses LayoutLMv3 + YOLOv8 ; Marker uses Surya OCR. All three are Python tools.
None are LuaJIT-native.

**Implementability : hard for in-process integration.** For ion7-rag we
should shell out to `marker` or `docling` as a CLI in an optional ingestion
pipeline, or stay on libmupdf / pdftotext as planned. **Roadmap, not v1.**

---

## 2. Embedding models

### 2.1 Qwen3-Embedding (June 2025) — clear v1 default

Apache-2.0, sizes 0.6B / 4B / 8B, 32K context, 100+ languages, MRL-truncatable,
instruction-aware. The 8B model held #1 on MTEB multilingual at 70.58
(June 2025). Paper :
[arXiv:2506.05176](https://arxiv.org/html/2506.05176v1). Official GGUFs :

- [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF)
- [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF)
- [Qwen/Qwen3-Embedding-8B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF)

**Implementability : trivial** — runs through ion7-core's existing encoder
bridge. Default embedder for v1.

### 2.2 Jina-embeddings-v4 (June 2025) and v5 (Feb 2026)

- **v4** ([arXiv:2506.18902](https://arxiv.org/abs/2506.18902)) : 3.8B,
  multimodal + multilingual, single- and multi-vector outputs. Strong on
  visually-rich docs.
  GGUFs : [jina-ai/jina-embeddings-v4-gguf](https://github.com/jina-ai/jina-embeddings-v4-gguf).
- **v5** (Feb 18 2026,
  [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small))
  : 677M small + 239M nano, built on Qwen3-0.6B-Base, distilled from
  Qwen3-Embedding-4B, task-specific LoRA adapters, MRL, 32K context, 119+
  languages, GGUF + MLX edge-quantized.

**Implementability : easy.** v5-nano (239M) is the natural default for
low-resource ion7-rag deployments ; v4 the path for visually-rich docs.

### 2.3 NVIDIA Llama-Embed-Nemotron-8B (early 2026)

Reported #1 on multilingual MTEB v2, fully open-weight. Community GGUFs are
surfacing but not yet officially from NVIDIA — verify before pinning.

### 2.4 EmbeddingGemma (2025)

Gemma-3 derived, 100+ languages, runs in <200 MB RAM with quantization, native
llama.cpp / Ollama support. **Implementability : easy.** Good "tiny default"
for embedded deployments.

### 2.5 nomic-embed-v2-MoE (Feb 2025) and Nomic Embed Multimodal (Apr 2025)

v2 is the first MoE general-purpose text embedder, Apache-2.0 ; Multimodal
(3B/7B) is Nomic's ColPali competitor. v1.5 was specifically trained for
binary-quantization compatibility. GGUF available via Ollama.
**Implementability : easy.**

### 2.6 Cohere Embed v4 (Apr 15 2025) and Voyage-3-large (Jan 7 2025)

Closed-weight. Embed v4 has Matryoshka + 128K context + multimodal + 65.2
MTEB ; Voyage-3-large has int8/binary QAT and Matryoshka.
**Not implementable locally** — treat as API-only fallback.

### 2.7 Matryoshka Representation Learning (MRL)

Kusupati et al., [arXiv:2205.13147](https://arxiv.org/abs/2205.13147). Now
table-stakes : most 2025+ embedders ship MRL training. Truncating a 1024-d
vector to 256-d gives a meaningful candidate-shortlist tier you can rerank
with the full 1024-d. Up to 14× speedups in retrieval.

**Implementability : trivial in SQLite.** Store the full vector once and
slice columns at query time, or store both 256-d and full vectors with the
small one queried first. **Bake into the ion7-rag vector schema.**

### 2.8 ColBERT / ColPali / ColQwen — late interaction

ColPali ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449), ICLR 2025) and
ColQwen2 treat PDF pages as images and produce per-patch ColBERT-style
multivectors. **2025 cost criticism :** storage explodes by ~T× (T tokens or
patches per doc). Mitigations :

- **ConstBERT** ([arXiv:2504.01818](https://arxiv.org/abs/2504.01818), 2025)
  — fixed-size multivector representation.
- **Token Pooling** ([arXiv:2409.14683](https://arxiv.org/html/2409.14683v1)).
- **HPC-ColPali** (SciTePress 2025) — K-Means quantization for 32×
  compression + attention-guided pruning.

**Implementability : hard.** sqlite-vec is single-vector-per-row by design.
Native late interaction would need flattened token vectors with
`(doc_id, token_idx)` keys and MaxSim done in SQL. **MUVERA (§7) is the
better path** — convert ColBERT vectors to single-vec FDEs and use
sqlite-vec normally.

---

## 3. Retrieval algorithms

### 3.1 Hybrid fusion : RRF, DBSF, Convex Combination

Reference paper : Bruch et al., *An Analysis of Fusion Functions for Hybrid
Retrieval*, ACM TOIS,
[arXiv:2210.11934](https://dl.acm.org/doi/10.1145/3596512). 2025 consensus :

- **RRF (Reciprocal Rank Fusion)** — zero-shot default, ignores scores
  entirely, parameter-light. Anthropic's 4:1 dense:BM25 ratio is RRF-style.
- **Convex Combination (CC)** — outperforms RRF in-domain and out-of-domain
  when ~tens of training queries are available. Requires score normalization
  but is normalization-agnostic in practice.
- **DBSF (Distribution-Based Score Fusion)** — z-score-normalize per-query at
  mean ± 3σ then sum. Query-adaptive without training data.
- **SRRF** (AutoRAG 2024) — RRF with a softmax over normalized scores.

**Implementability : easy in pure Lua.** **Ship RRF as default, DBSF as
opt-in, CC as a knob with a tuning helper.**

### 3.2 HyPE — ingest-time HyDE (2025)

Vanilla HyDE generates a hypothetical answer at *query time*. **HyPE**
(Hypothetical Prompt Embeddings) flips it : at *ingest time*, generate
hypothetical questions per chunk and embed those alongside the chunk.
Reports +42 ppt precision and +45 ppt recall on some datasets vs vanilla
retrieval. Eliminates per-query latency. Composes with Contextual Retrieval
(do both — context for grounding, HyPE for question-shaped vectors).

**Implementability : easy.** ion7-llm.Engine + ion7-grammar can constrain the
generator to "produce 3 questions" via JSON schema. Store as separate rows
in sqlite-vec with a `chunk_id` foreign key. **Recommended for v1.**

### 3.3 Adaptive query routing (2025-2026)

Adaptive-RAG (Jeong et al. 2024) trained a small T5 to route queries to
no-retrieval / single-hop / multi-hop. 2025-2026 extensions :

- **RAGRouter-Bench** ([arXiv:2604.03455](https://arxiv.org/abs/2604.03455),
  2026) — a TF-IDF + SVM router achieves macro-F1 0.928 with 28 % token
  savings vs always-expensive baseline. **Lexical features beat sentence
  embeddings by 3.1 F1** — a counterintuitive but reproducible finding.
- **Tier-Based Adaptive Query Routing**
  ([arXiv:2604.14222](https://arxiv.org/html/2604.14222)) — vertical-specific
  (finance, legal, medical).
- Production benchmarks (Jan 2026) report 30-40 % latency reduction with
  accuracy improved.

**Implementability : easy.** Pure-Lua TF-IDF + small classifier ; alternatively
a small-LLM zero-shot router using ion7-grammar JSON schema. **Recommended
for v1.**

### 3.4 GraphRAG / LightRAG / HippoRAG 2 / FastGraphRAG

- **LightRAG** (HKU,
  [EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.568.pdf)) : KG +
  embedding hybrid ; ~30 % lower latency than naive RAG ; 2025-2026 added
  reranker support, RAG-Anything multimodal mode, Langfuse tracing,
  OpenSearch backend.
- **HippoRAG 2 / *From RAG to Memory*** — Gutiérrez et al.,
  [arXiv:2502.14802](https://icml.cc/virtual/2025/poster/45585), ICML 2025.
  Personalized PageRank over an LLM-extracted KG ; +7 % on associative-memory
  tasks vs SOTA embedder ; *less* offline-indexing cost than GraphRAG /
  RAPTOR / LightRAG. Likely the strongest 2025 graph-RAG result.
- **nano-graphrag / FastGraphRAG** — lightweight ports. FastGraphRAG noted
  to fail on local 7-8B models : the LLM cannot reliably output the
  structured-JSON KG even with retries.
- **E²-GraphRAG** ([arXiv:2505.24226](https://arxiv.org/abs/2505.24226), 2025)
  — streamlined graph-based RAG.
- **Graph-R1** ([arXiv:2507.21892](https://arxiv.org/abs/2507.21892), 2025) —
  end-to-end RL agentic GraphRAG.
- **ROGRAG** (ACL 2025 Demo) — robustly-optimized GraphRAG.

**2025 critique of vanilla GraphRAG :** more moving parts, slower at
sub-second latency budgets ; without a schema the graph becomes a "noisy
hairball." ROI in research / legal / medical, but one-off use cases don't
justify setup cost.

**Implementability for ion7-rag : moderate.** KG extraction with
`ion7-grammar`'s JSON-Schema → GBNF + a Pool worker is a strong fit — this
is exactly where ion7-grammar shines : guaranteed-valid JSON triples from
local 7-8B models. Personalized PageRank in pure Lua over an SQLite-stored
adjacency list is straightforward. **HippoRAG-2-style is the right v1.5
graph approach** ; defer past v1.

### 3.5 Search-R1 and RL-trained agentic retrieval (Mar 2025)

Search-R1 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516), Jin et al.
Mar 12 2025) — RL-trains an LLM to interleave reasoning and search-engine
calls ; +41 % over RAG baselines on Qwen2.5-7B across 7 QA datasets. Uses
retrieved-token masking for stable PPO/GRPO. *A Survey on Reasoning Agentic
RAG* (ACL findings IJCNLP 2025) catalogs the space.

**Implementability : moderate at inference time** (we don't train, we run a
pre-trained Search-R1 checkpoint via ion7-llm with a tool-call grammar).
**Training is out of scope.** **Not v1 ; design ion7-rag's tool-call API so
a Search-R1-style model can plug in later.**

### 3.6 TAG / Table-Augmented Generation

TAG (Biswal et al., 2024) frames structured-data QA as
NL→SQL → execution → answer. **TableRAG**
([arXiv:2506.10380](https://arxiv.org/html/2506.10380v1), 2025) extends to
heterogeneous-doc reasoning. **MT-RAIG** (2025) benchmarks multi-table insight
generation.

**Implementability : easy** *and* a natural fit for SQLite. ion7-grammar can
constrain SQL generation against a known schema. **Worth shipping as
`ion7-rag.tag` submodule for v1 or v1.1.**

### 3.7 Long-context vs RAG (2025 debate)

- *Retrieval Augmented Generation or Long-Context LLMs ?* — Google DeepMind,
  EMNLP 2024.
- *Long Context vs. RAG for LLMs : An Evaluation and Revisits* —
  [arXiv:2501.01880](https://arxiv.org/abs/2501.01880), Jan 2025. Long
  context generally outperforms RAG on Wikipedia QA, but summarization-based
  retrieval is comparable.
- **LaRA** ([ICML 2025](https://openreview.net/forum?id=CLF25dahgA)) —
  2326 test cases × 11 LLMs : "no silver bullet."

Consensus : RAG is **8-82× cheaper** in tokens and faster in latency than
long-context. Long context wins quality when budget is unconstrained.

**Implication for ion7-rag :** RAG is alive ; no design pivot needed. Ship a
"long-context bypass" mode that, when the corpus fits, just stuffs the model
— measure and choose at runtime.

---

## 4. Reranking and post-retrieval

### 4.1 Cross-encoders with GGUF

- **bge-reranker-v2.5-gemma2-lightweight** — gemma-2-9b base, supports token
  compression and layerwise lightweight ops.
  [bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF)
  for v2-m3 ; lightweight variants surfacing.
- **Qwen3-Reranker** (June 2025,
  [Qwen3-Reranker collection](https://huggingface.co/collections/Qwen/qwen3-reranker))
  : 0.6B / 4B / 8B ; +3.0 points 0.6B → 8B on retrieval tasks ; Apache-2.0,
  multilingual.
- **mxbai-rerank-v2** (Mar 2025,
  [mixedbread-ai/mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2)
  ; [blog](https://www.mixedbread.com/blog/mxbai-rerank-v2)) : base 0.5B +
  large 1.5B ; trained with GRPO + contrastive + preference learning ; 100+
  languages ; 8K (32K-compatible) context ; outperforms Cohere/Voyage on
  BEIR (NDCG@10 = 57.49). Apache-2.0. Quantized GGUFs on community accounts.
- **jina-reranker-v3** ([model page](https://jina.ai/models/jina-reranker-v3/),
  Oct 1 2025) : 0.6B parameter listwise reranker, "last but not late
  interaction" — single context window over query + all candidates.

**Implementability : easy via ion7-core.** A reranker is an encoder model
with a regression head ; the bridge already supports classifier-style hooks.
**Ship Qwen3-Reranker-0.6B as default ; mxbai-rerank-base-v2 as a strong
alternative.**

### 4.2 LLM-as-listwise-reranker

- **RankZephyr** (Mistral-7B base) — listwise SOTA pre-2025, baseline.
- **Rank-K** ([arXiv:2505.14432](https://arxiv.org/html/2505.14432v1), May
  2025) — test-time reasoning listwise on QwQ-32B base ; beats RankZephyr on
  TREC DL and NeuCLIR.
- **SETwise + heapsort reproducibility study** — SIGIR 2025
  ([doi 10.1145/3726302.3730338](https://doi.org/10.1145/3726302.3730338)).
- **Guiding Retrieval using LLM-based Listwise Rankers** —
  [arXiv:2501.09186](https://arxiv.org/abs/2501.09186).
- **How Good are LLM-based Rerankers ?** —
  [arXiv:2508.16757](https://arxiv.org/abs/2508.16757), Aug 2025 : warns
  that listwise can be brittle on harder distributions.

**Implementability : easy with ion7-grammar** (constrain to a permutation-of-N
JSON schema). SETwise heapsort fits ion7-llm.Pool naturally — many short
calls. **Optional v1 feature : `ion7-rag.rerank.listwise`.**

### 4.3 Provence (ICLR 2025) and XProvence (2026)

- **Provence** — Chirkova et al.,
  [arXiv:2501.16214](https://arxiv.org/abs/2501.16214), ICLR 2025. DeBERTa-based
  context pruner that does token-level sequence labeling + reranking jointly.
  "Almost zero-cost" pruning with negligible quality drop on NQ / TyDi /
  PopQA / HotpotQA / BioASQ.
- **XProvence** ([arXiv:2601.18886](https://arxiv.org/abs/2601.18886), 2026)
  — zero-cost multilingual extension, 16 languages trained, 100+ supported.

**Implementability : moderate.** DeBERTa is encoder-only and runs on
llama.cpp's encoder path. The token-classification head needs a small bridge
addition. The payoff is large — context pruning shrinks generator-time cost
significantly. **Recommended for v1 or v1.1.**

### 4.4 LongLLMLingua and FiD evolutions

LongLLMLingua compresses long context via per-token importance. Less
attention in 2025-2026 ; Provence subsumes its use case for RAG specifically.
Fusion-in-Decoder is now niche relative to long-context generators.

---

## 5. Generation-time RAG

### 5.1 Self-RAG, CRAG, FLARE — current status

A 2025 benchmark on 250 clinical vignettes ranked them :

- **Self-RAG** — 5.8 % hallucination, lowest of 12 RAG variants.
- **CRAG** (Corrective RAG) — 10.5 % hallucination, P@5=0.69, 240 ms latency
  ; "most practical first step into Agentic RAG".
- **FLARE** — competitive but slower than CRAG.

All three remain implementable patterns ; all three are now sub-architectures
inside Search-R1-style RL agents.

**Implementability : easy as an orchestration layer over ion7-llm.** Self-RAG's
reflection tokens fit ion7-grammar perfectly (constrain output to
`{"retrieve": bool, "relevant": bool, "supported": bool, ...}`). **Ship CRAG
as v1 default agentic loop ; Self-RAG-style reflection as opt-in.**

### 5.2 Speculative-RAG (ICLR 2025)

Wang et al., [arXiv:2407.08223](https://arxiv.org/abs/2407.08223). A small
specialist LM produces *parallel drafts* from disjoint subsets of retrieved
docs ; a large generalist verifies. +12.97 % accuracy / -50.83 % latency on
PubHealth.

**Implementability : moderate, good fit.** ion7-llm.Pool can run draft
workers in parallel ; ion7-core has speculative-decoding plumbing for
token-level. Document-level speculation is a different orchestration layer
but fits the Pool model. **Strong candidate for v1.5.**

### 5.3 InstructRAG, FAIR-RAG, Self-Correcting RAG

- **FAIR-RAG** ([arXiv:2510.22344](https://arxiv.org/abs/2510.22344), Oct
  2025) — faithful adaptive iterative refinement.
- **Self-Correcting RAG** ([arXiv:2604.10734](https://arxiv.org/abs/2604.10734),
  2026) — MMKP context selection + NLI-guided MCTS ; heavyweight.
- **Search-R1 / Agentic-RAG-R1** — see § 3.5.

**Implementability :** iterative-refinement is just a control loop on top of
ion7-llm + ion7-rag.retrieve. Easy. The research contribution is the policy.

### 5.4 Thinking-mode RAG (2026)

NVIDIA Nemotron-3 ships a `thinking_budget` (recommended 8192 tokens) and
emits reasoning tokens before the final answer. 2025 survey *Synergizing RAG
and Reasoning* ([arXiv:2504.15909](https://arxiv.org/html/2504.15909v1)) and
Hyper-RAG (hypergraph-driven, 2026) show measurable gains when the model
reasons over retrieved evidence before answering.

**Implementability : easy.** "Thinking" is just unconstrained text before a
grammar-constrained final answer. **Worth a docs-level pattern, not a
separate module.**

### 5.5 Industry-reported failure mode (2026)

Multiple production analyses converge on : **"73 % of RAG failures are
retrieval failures, not generation failures."** Implication for ion7-rag's
budget : prioritize retrieval (Contextual Retrieval, hybrid fusion, reranker,
query routing) over generation-time tricks.

---

## 6. Evaluation

### 6.1 RAGAs

Es et al., [arXiv:2309.15217](https://arxiv.org/abs/2309.15217) (last revised
Apr 28 2025). Reference-free RAG metric framework. Metrics : Context
Precision, Context Recall, Faithfulness, Answer Relevancy, Response
Groundedness. All metrics are pure prompts → portable to ion7-llm directly.

**Implementability : easy.** Port the metric prompts to LuaJIT + ion7-grammar
to constrain the judge's outputs. **Ship as `ion7-rag.eval`.**

### 6.2 MTEB v2 / MMTEB (2025)

- **MMTEB** ([arXiv:2502.13595](https://arxiv.org/abs/2502.13595), Feb 2025)
  — 500+ tasks, 250+ languages.
- **MTEB v2** ([announcement](https://huggingface.co/blog/isaacchung/mteb-v2),
  Chung et al., 2025) — refactor for long-term reproducibility, multimodal
  default ; MIEB image extension.

**Implication :** when validating an embedder choice for ion7-rag, point at
MMTEB.

### 6.3 TREC RAG 2024/2025

[trec-rag.github.io](https://trec-rag.github.io/). Retrieval (R), Augmented
Generation (AG), and joint RAG tasks on MS MARCO Segment v2.1.
Ragnarök baselines (TREC 2024).

### 6.4 Hallucination detection

- **Lynx-8B / Lynx-70B** — Patronus AI,
  [arXiv:2407.08488](https://arxiv.org/html/2407.08488v1), late 2024.
  Open-source Llama-3 finetune ; beats GPT-4 on RAG hallucination tasks.
- **HaluBench** — 15K examples across domains.
- *Real-Time Evaluation Models for RAG* —
  [arXiv:2503.21157](https://arxiv.org/abs/2503.21157), Mar 2025.
- **Vectara HHEM** — alternative.

**Implementability : easy.** Lynx-8B is a Llama-3 finetune → runs natively on
ion7-core. **Ship a Lynx-8B-based hallucination check as opt-in.**

---

## 7. Indexing and store innovations

### 7.1 sqlite-vec (Alex Garcia, v0.1.0 Aug 2024, active through 2026)

Capabilities directly relevant to ion7-rag :

- **Binary quantization** (`vec_quantize_binary`) : 1 bit/dim, 32×
  compression. Hamming distance via XOR + popcount. Models trained for it :
  nomic-embed-text-v1.5, mxbai-embed-large-v1, Voyage-3-large, Cohere Embed
  v4. [Guide](https://alexgarcia.xyz/sqlite-vec/guides/binary-quant.html).
- **Scalar quantization** (`vec_quantize_int8`) : 4-8× compression with
  minor quality loss.
  [Guide](https://alexgarcia.xyz/sqlite-vec/guides/scalar-quant.html).
- **MRL truncation :** store the full vector once, slice columns at query
  time.

The ["Local-First RAG : Vector Search in SQLite with Hamming Distance"](https://www.sitepoint.com/local-first-rag-vector-search-in-sqlite-with-hamming-distance/)
workflow is exactly ion7-rag's target.

**Implementability : trivial via lsqlite3.** Use binary quantization for the
candidate-shortlist tier ; full float32 (or int8) for the rerank tier.

### 7.2 MUVERA (Google Research, June 2025)

[Blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/),
[paper](https://research.google/pubs/muvera-multi-vector-retrieval-via-fixed-dimensional-encodings/).
Constructs **Fixed Dimensional Encodings (FDEs)** : the inner product of two
FDEs approximates the multi-vector ChamferSim/MaxSim score, so multi-vector
retrieval reduces to single-vec MIPS. **+10 % recall, -90 % latency** vs prior
SOTA on BEIR.

**Implementability : hard but high-value.** The FDE construction is a
deterministic projection — implementable in pure Lua / FFI. Once we have FDE
vectors, sqlite-vec works normally. **This is the path to ColBERT/ColPali
support without breaking sqlite-vec's single-vec assumption.** Strong
candidate for v1.5.

### 7.3 DiskANN / Vamana / SPANN

DiskANN (Microsoft Vamana graph) — billion-scale, 95 % recall at 5 ms,
~5 % RAM overhead, rest on NVMe. Now in SQL Server 2025. **SPANN** uses
~30 % RAM, 2× faster than DiskANN above that threshold. **B+ANN**
([arXiv:2511.15557](https://arxiv.org/abs/2511.15557), late 2025),
**FAST-PipeANN** (2026).

**Implementability : hard inside SQLite.** sqlite-vec currently uses
brute-force / IVF ; HNSW/DiskANN integration is roadmap-level for sqlite-vec
itself. For ion7-rag's "single-machine, embedded" target, brute-force on
binary-quantized vectors handles ~10M vectors comfortably. **Defer ANN until
corpus exceeds that.**

### 7.4 Per-corpus sizing rule of thumb (2025-2026 consensus)

- **< 100K chunks** — brute-force float32 sqlite-vec.
- **100K-10M chunks** — brute-force binary-quantized + float32 rerank tier.
- **10M-100M chunks** — needs HNSW / IVF (out of v1 scope).
- **100M+** — DiskANN / cloud territory.

---

## 8. What's overrated or didn't pan out

- **Pure semantic chunking** — multiple 2025 benchmarks (NAACL 2025 Findings,
  FloTorch 2026) show fixed 256-512-token recursive splitting matches or
  beats it. Don't make it the default.
- **Vanilla GraphRAG (Microsoft, 2024)** — high indexing cost, requires a
  schema, brittle without one. HippoRAG 2 explicitly outperforms with much
  lower offline cost. Don't ship GraphRAG-style ; ship HippoRAG-2-style if
  doing graph-RAG at all.
- **FastGraphRAG with local 7-8B** — fails because the LLM cannot output
  well-formed structured-JSON KG triples even with retries. (ion7-grammar
  fixes this — guaranteed valid JSON. Worth noting as a competitive
  advantage of the ion7 stack.)
- **Multi-vector ColBERT-style at storage cost** — universally flagged as
  expensive. MUVERA and ConstBERT exist precisely because of this. Don't
  store T-vectors-per-doc naively.
- **"Long context killed RAG"** — empirically false. RAG is 8-82× cheaper,
  faster, and within quality range. The narrative has flipped to "retrieval
  focuses attention where it matters."
- **Listwise LLM rerankers as default** — SIGIR 2025 reproducibility study
  and arXiv:2508.16757 caution they're brittle out-of-distribution. Use
  cross-encoders (Qwen3-Reranker, mxbai-rerank-v2) as default ; listwise as
  opt-in.
- **Pure HyDE (query-time)** — adds latency every query. **HyPE (ingest-time)
  wins** for production.
- **Vanilla Self-RAG with reflection tokens** — lowest hallucination but high
  latency ; CRAG is the more practical first step.

---

## 9. v1 shortlist (ranked)

1. **Recursive 512-token splitter, 15-25 % overlap, 150-token min-chunk
   floor (LPeg).** Empirically-validated 2026 default ; avoid the
   semantic-chunking trap.
2. **Anthropic Contextual Retrieval, formalized per arXiv:2504.19754
   (ECIR 2025).** Single biggest retrieval-quality win per dollar. Use a
   *small* fast contextualizer (0.6B-1B) — `ion7-llm.Pool` with a draft
   model worker and RadixAttention exact-match prefix cache make this
   essentially free per chunk after the first.
3. **Hybrid retrieval : sqlite-vec dense + FTS5 BM25, fused with RRF
   (default), DBSF and Convex Combination as opt-in.** The 4:1 dense:BM25
   weighting from Anthropic is a sane prior.
4. **Qwen3-Embedding-0.6B as default ; Jina-v5-nano (239M) for tiny
   deployments ; Qwen3-Embedding-8B as the "quality" tier.** All
   Apache-2.0, MRL, 32K context, GGUF, multilingual. Schema-store binary-
   quantized (shortlist tier) and full int8 (rerank tier).
5. **Qwen3-Reranker-0.6B (default) or mxbai-rerank-base-v2 (alternative)
   as the cross-encoder rerank stage.** 2025-trained rerankers measurably
   beat 2024 BGE on multilingual.
6. **Late chunking (Jina arXiv:2409.04701 v3) as opt-in chunker** for docs
   under the embedder's 32K context. Complements Contextual Retrieval ;
   cheaper at ingest. Requires bridge work for per-token embeddings.
7. **HyPE (ingest-time hypothetical questions) as opt-in retrieval booster.**
   Constrained-JSON generation via ion7-grammar makes this trivial. Pairs
   cleanly with Contextual Retrieval.
8. **Adaptive query routing (TF-IDF + tiny classifier baseline,
   RAGRouter-Bench 2026).** ~28 % token savings with no quality loss on
   typical mixed workloads. Lexical features beat sentence embeddings here
   — counterintuitive but reproducible.
9. **CRAG-style retrieval-relevance gate as default agentic loop ; Self-RAG
   reflection (ion7-grammar JSON) as opt-in.** Best practical
   hallucination-rate-per-latency from the 2025 clinical-vignette benchmark.
10. **`ion7-rag.eval` module : RAGAs metrics + Lynx-8B GGUF for
    hallucination scoring.** Native to ion7-core ; lets users self-evaluate
    without leaving the stack.

**Deferred past v1.** RAPTOR trees ; HippoRAG-2-style graph (the
ion7-grammar advantage on local KG extraction is a v1.5 differentiator) ;
Speculative-RAG draft pool ; Provence / XProvence pruning ; MUVERA + ColPali
multimodal path ; Search-R1-style RL agentic loops ; layout-aware Docling /
Marker integration.

---

## 10. ion7-specific differentiators

Three places where the ion7 stack lets us do things the Python-RAG ecosystem
struggles with :

1. **`ion7-grammar` resolves FastGraphRAG's failure mode.** The known
   blocker (local 7-8B models cannot reliably emit valid KG-triple JSON
   even with retries) disappears entirely under GBNF — output validity is
   guaranteed by construction, not by sampling luck. This makes a v1.5
   HippoRAG-2-style graph layer realistic on commodity local models.
2. **`ion7-llm.Pool` + RadixAttention exact-match cache aligns with
   Contextual Retrieval's cost profile.** The full document is shared as a
   prefix across all its chunks during contextualization — exactly what
   the prompt cache amortizes. We get Anthropic's cost story for free.
3. **`ion7-grammar` makes constrained listwise rerankers and Self-RAG
   reflection tokens cheap.** Permutation-of-N for listwise, structured
   reflection JSON for Self-RAG, structured tool-call JSON for Search-R1
   — all are GBNF one-liners, no retries, no parser-tolerance heuristics.

---

## 11. Sources

### Primary papers (chronological)

- *Matryoshka Representation Learning* — Kusupati et al.,
  [arXiv:2205.13147](https://arxiv.org/abs/2205.13147).
- *RAGAs : Automated Evaluation of Retrieval Augmented Generation* — Es
  et al., [arXiv:2309.15217](https://arxiv.org/abs/2309.15217).
- *ColPali : Efficient Document Retrieval with Vision Language Models* —
  [arXiv:2407.01449](https://arxiv.org/abs/2407.01449).
- *Speculative RAG* — Wang et al.,
  [arXiv:2407.08223](https://arxiv.org/abs/2407.08223).
- *Lynx : An Open Source Hallucination Evaluation Model* —
  [arXiv:2407.08488](https://arxiv.org/html/2407.08488v1).
- *Late Chunking* — Günther et al. (Jina),
  [arXiv:2409.04701](https://arxiv.org/abs/2409.04701) (v3, Jul 2025).
- *Token Pooling for ColBERT* —
  [arXiv:2409.14683](https://arxiv.org/html/2409.14683v1).
- *Meta-Chunking* — Zhao et al.,
  [arXiv:2410.12788](https://arxiv.org/abs/2410.12788), ICLR 2025.
- *Long Context vs RAG : An Evaluation and Revisits* —
  [arXiv:2501.01880](https://arxiv.org/abs/2501.01880), Jan 2025.
- *Guiding Retrieval using LLM-based Listwise Rankers* —
  [arXiv:2501.09186](https://arxiv.org/abs/2501.09186).
- *Provence* — Chirkova et al.,
  [arXiv:2501.16214](https://arxiv.org/abs/2501.16214), ICLR 2025.
- *MMTEB : Massive Multilingual Text Embedding Benchmark* —
  [arXiv:2502.13595](https://arxiv.org/abs/2502.13595).
- *HippoRAG 2 / From RAG to Memory* — Gutiérrez et al.,
  [ICML 2025](https://icml.cc/virtual/2025/poster/45585).
- *Search-R1* — Jin et al.,
  [arXiv:2503.09516](https://arxiv.org/abs/2503.09516), Mar 2025.
- *Real-Time Evaluation Models for RAG* —
  [arXiv:2503.21157](https://arxiv.org/abs/2503.21157), Mar 2025.
- *ConstBERT : Constant-Space Multi-Vector Retrieval* —
  [arXiv:2504.01818](https://arxiv.org/abs/2504.01818).
- *Synergizing RAG and Reasoning* —
  [arXiv:2504.15909](https://arxiv.org/html/2504.15909v1).
- *Reconstructing Context : Evaluating Advanced Chunking Strategies for RAG*
  — Pham et al., [arXiv:2504.19754](https://arxiv.org/abs/2504.19754),
  ECIR 2025.
- *Rank-K* — [arXiv:2505.14432](https://arxiv.org/html/2505.14432v1).
- *E²-GraphRAG* — [arXiv:2505.24226](https://arxiv.org/abs/2505.24226).
- *Qwen3-Embedding* — [arXiv:2506.05176](https://arxiv.org/html/2506.05176v1),
  June 2025.
- *TableRAG* — [arXiv:2506.10380](https://arxiv.org/html/2506.10380v1).
- *Jina-embeddings-v4* —
  [arXiv:2506.18902](https://arxiv.org/abs/2506.18902).
- *Graph-R1* — [arXiv:2507.21892](https://arxiv.org/abs/2507.21892).
- *How Good are LLM-based Rerankers ?* —
  [arXiv:2508.16757](https://arxiv.org/abs/2508.16757), Aug 2025.
- *FAIR-RAG* — [arXiv:2510.22344](https://arxiv.org/abs/2510.22344),
  Oct 2025.
- *B+ANN* — [arXiv:2511.15557](https://arxiv.org/abs/2511.15557).
- *XProvence* — [arXiv:2601.18886](https://arxiv.org/abs/2601.18886), 2026.
- *Self-Correcting RAG* —
  [arXiv:2604.10734](https://arxiv.org/abs/2604.10734), 2026.
- *RAGRouter-Bench* —
  [arXiv:2604.03455](https://arxiv.org/abs/2604.03455), 2026.
- *Tier-Based Adaptive Query Routing* —
  [arXiv:2604.14222](https://arxiv.org/html/2604.14222), 2026.

### Industry posts and tool docs

- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval).
- [MUVERA — Google Research blog](https://research.google/blog/muvera-making-multi-vector-retrieval-as-fast-as-single-vector-search/)
  + [paper](https://research.google/pubs/muvera-multi-vector-retrieval-via-fixed-dimensional-encodings/).
- [Mixedbread — mxbai-rerank-v2](https://www.mixedbread.com/blog/mxbai-rerank-v2).
- [Nomic Embed Multimodal](https://www.nomic.ai/news/nomic-embed-multimodal),
  Apr 2025.
- [Jina-reranker-v3](https://jina.ai/models/jina-reranker-v3/), Oct 2025.
- [Cohere Embed v4 changelog](https://docs.cohere.com/changelog/embed-multimodal-v4),
  Apr 2025.
- [Voyage-3-large](https://blog.voyageai.com/2025/01/07/voyage-3-large/),
  Jan 2025.
- [MTEB v2](https://huggingface.co/blog/isaacchung/mteb-v2).
- [TREC RAG track](https://trec-rag.github.io/).
- [sqlite-vec — binary quantization guide](https://alexgarcia.xyz/sqlite-vec/guides/binary-quant.html).
- [sqlite-vec — scalar quantization guide](https://alexgarcia.xyz/sqlite-vec/guides/scalar-quant.html).
- [Local-First RAG with SQLite and Hamming Distance — SitePoint](https://www.sitepoint.com/local-first-rag-vector-search-in-sqlite-with-hamming-distance/).
- [LightRAG — EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.568.pdf).

### Model cards and weights

- [Qwen/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B-GGUF).
- [Qwen/Qwen3-Embedding-4B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-4B-GGUF).
- [Qwen/Qwen3-Embedding-8B-GGUF](https://huggingface.co/Qwen/Qwen3-Embedding-8B-GGUF).
- [Qwen3-Reranker collection](https://huggingface.co/collections/Qwen/qwen3-reranker).
- [jina-ai/jina-embeddings-v4-gguf](https://github.com/jina-ai/jina-embeddings-v4-gguf).
- [jinaai/jina-embeddings-v5-text-small](https://huggingface.co/jinaai/jina-embeddings-v5-text-small).
- [mixedbread-ai/mxbai-rerank-large-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v2).
- [nomic-ai/nomic-embed-text-v2-moe](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe).
- [gpustack/bge-reranker-v2-m3-GGUF](https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF).
