--- @module ion7.rag.chunk.late
--- @author  ion7 / Ion7 Project Contributors
---
--- Late chunking — Günther et al., arXiv:2409.04701 (v3 Jul 2025).
---
--- Chunked embedding normally embeds each chunk in isolation, dropping
--- the surrounding context. Late chunking inverts the order : decode
--- the WHOLE document through a long-context embedder with
--- `pooling = "none"` to get one embedding per token, then, per chunk,
--- mean-pool the embeddings of the tokens that fall inside the chunk's
--- character span. Each chunk vector then carries contextual signal
--- from the whole document without any LLM call — complementary to
--- Anthropic Contextual Retrieval, and cheaper at ingest.
---
--- Requirements :
---   - Embedder Context built with `pooling = "none"` and an `n_ctx`
---     large enough to hold the full document at once.
---   - ion7-core's `Context:embedding_token_ptr` and
---     `Context:decode_for_embeddings` (Context API).
---
--- Character-to-token alignment uses cumulative-prefix tokenisation :
--- for each chunk's `char_end`, the prefix `text[0..char_end]` is
--- tokenised and the resulting token count is recorded as the chunk's
--- exclusive token boundary. Off-by-one drift around chunk boundaries
--- sits well below the noise floor of a 1024-d embedder ; the
--- `boundaries` opt is exposed so a quality-conscious caller can
--- override the alignment.

local M = {}

-- ── Helpers ─────────────────────────────────────────────────────────────

--- Compute exclusive token-end positions for each chunk by
--- re-tokenising text prefixes. Returns an array where `boundaries[i]`
--- is the exclusive end token index, in the full-doc token stream, of
--- chunk `i`.
local function _chunk_token_boundaries(vocab, full_text, chunks)
    local boundaries = {}
    for i, c in ipairs(chunks) do
        local prefix = full_text:sub(1, c.char_end)
        -- `add_bos = false` and `parse_special = true` match the call
        -- that produced the full-doc tokens during decode, so the
        -- prefix counts line up with positions in the decoded stream.
        local _, n = vocab:tokenize(prefix, false, true)
        boundaries[i] = tonumber(n)
    end
    return boundaries
end

local function _mean_pool(ctx, lo, hi, dim)
    -- Mean-pool token embeddings in [lo, hi) (0-indexed, exclusive end).
    local first = ctx:embedding_token_ptr(lo)
    if first == nil then return nil end

    local acc = {}
    for d = 0, dim - 1 do acc[d + 1] = first[d] end

    local n = 1
    for tok = lo + 1, hi - 1 do
        local p = ctx:embedding_token_ptr(tok)
        if p ~= nil then
            for d = 0, dim - 1 do
                acc[d + 1] = acc[d + 1] + p[d]
            end
            n = n + 1
        end
    end

    if n > 1 then
        local inv = 1 / n
        for d = 1, dim do acc[d] = acc[d] * inv end
    end
    return acc
end

-- ── Public API ──────────────────────────────────────────────────────────

--- Encode each chunk's vector via late chunking on the full document.
---
--- @param  opts table {
---     ctx       ion7.core.Context  REQUIRED. Built with pooling = "none"
---                  and an n_ctx large enough to hold the full document
---                  tokens.
---     vocab     ion7.core.Vocab    REQUIRED. Same vocab as the model that
---                  backs `ctx`.
---     doc_text  string             REQUIRED. The full document text.
---     chunks    Chunk[]            REQUIRED. Each carries `char_end`
---                  pointing into `doc_text` (output of `recursive.chunk`
---                  is consumed as-is).
---     dim       integer?           Embedding dimension. Defaults to the
---                  Context's parent Model n_embd when reachable.
---     boundaries integer[]?        Optional override of the per-chunk
---                  exclusive token-end positions. When omitted, recomputed
---                  via cumulative prefix tokenisation (correct but
---                  O(n_chunks × len_prefix)).
--- }
--- @return number[][]  One embedding per input chunk, in input order.
--- @raise When required opts are missing, when `dim` cannot be resolved,
---        or when the document tokenises beyond `ctx:n_ctx()`.
function M.encode(opts)
    local ctx       = assert(opts.ctx,      "late.encode : opts.ctx required")
    local vocab     = assert(opts.vocab,    "late.encode : opts.vocab required")
    local doc_text  = assert(opts.doc_text, "late.encode : opts.doc_text required")
    local chunks    = assert(opts.chunks,   "late.encode : opts.chunks required")

    if #chunks == 0 then return {} end

    local dim = opts.dim
    if not dim and ctx._model_ref then
        dim = tonumber(ctx._model_ref:n_embd())
    end
    assert(dim, "late.encode : opts.dim required (no model back-ref on ctx)")

    -- Single full-doc decode. `decode_for_embeddings` sets
    -- `logits[i] = 1` on every token, which is required for llama.cpp
    -- to emit a per-token embedding. A plain `decode` only flags the
    -- last token and leaves the rest at zero, silently producing
    -- all-zero per-token vectors that would mean-pool to garbage.
    local toks, n_tokens = vocab:tokenize(doc_text, false, true)
    if tonumber(n_tokens) > ctx:n_ctx() then
        error(string.format(
            "ion7.rag.chunk.late : doc tokenises to %d tokens but ctx n_ctx is %d. " ..
            "Either grow the context, or chunk the doc first and run late chunking " ..
            "per super-chunk.", tonumber(n_tokens), ctx:n_ctx()), 2)
    end
    ctx:kv_clear()
    ctx:decode_for_embeddings(toks, n_tokens)

    -- Char→token boundaries (cumulative prefix re-tokenisation).
    local boundaries = opts.boundaries
                    or _chunk_token_boundaries(vocab, doc_text, chunks)

    local prev_end = 0
    local out = {}
    for i, c in ipairs(chunks) do
        local hi = math.min(boundaries[i], tonumber(n_tokens))
        local lo = math.max(prev_end, 0)
        if hi <= lo then hi = lo + 1 end -- guard tiny / degenerate chunks
        out[i] = _mean_pool(ctx, lo, hi, dim) or {}
        prev_end = hi
    end
    return out
end

return M
