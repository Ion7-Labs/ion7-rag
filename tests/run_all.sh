#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Run the full ion7-rag test suite.
#
# Files starting with `0` are pure-Lua + module-load tests (no model, no
# corpus). Files starting with `1` need the SQLite substrate. Files
# starting with `2` need the corpus loaders. Files starting with `3` need
# fusion / retrieval glue. Files starting with `4`+ need real models
# (ION7_MODEL, ION7_EMBED_MODEL, ION7_RERANK_MODEL).
#
# Discovery is alphabetic, so the prefix dictates ordering.
#
# Usage :
#   bash tests/run_all.sh
#   ION7_EMBED_MODEL=/path/embed.gguf bash tests/run_all.sh
#
# Optional environment :
#   ION7_MODEL           Chat model (.gguf) for tests that drive an LLM.
#   ION7_EMBED_MODEL     Embedder model (.gguf).
#   ION7_RERANK_MODEL    Cross-encoder reranker (.gguf).
#   ION7_CONTEXT_MODEL   Small contextualizer (.gguf), defaults to ION7_MODEL.
#   ION7_RAG_CORPUS      Path to a directory of test documents.
#   ION7_GPU_LAYERS      Override n_gpu_layers (default 0 — pure CPU).
#   ION7_CORE_SRC        Override ion7-core source root.
#   ION7_LLM_SRC         Override ion7-llm source root.
#   ION7_GRAMMAR_SRC     Override ion7-grammar source root.
#   ION7_LIBLLAMA_PATH   Pin a specific libllama.so for ion7-core's FFI loader.
#   ION7_LIBGGML_PATH    Pin a specific libggml.so.
#   ION7_BRIDGE_PATH     Pin a specific ion7_bridge.so.
#   ION7_SKIP            Whitespace-separated list of file basenames to skip.
# ──────────────────────────────────────────────────────────────────────────

set -e

PASS=0
FAIL=0
SKIP=0

cd "$(dirname "$0")/.."

run_suite() {
    local file="$1"
    local name
    name="$(basename "$file" .lua)"

    if [ -n "$ION7_SKIP" ] && echo "$ION7_SKIP" | grep -qw "$(basename "$file")"; then
        printf "\n\033[33m══ SKIP %-40s\033[0m\n" "$name (in ION7_SKIP)"
        SKIP=$((SKIP + 1))
        return
    fi

    printf "\n\033[1m══ %-40s \033[0m%s\n" "$name" "$(printf '═%.0s' {1..18})"
    if ION7_MODEL="$ION7_MODEL" \
       ION7_EMBED_MODEL="$ION7_EMBED_MODEL" \
       ION7_RERANK_MODEL="$ION7_RERANK_MODEL" \
       ION7_CONTEXT_MODEL="$ION7_CONTEXT_MODEL" \
       ION7_RAG_CORPUS="$ION7_RAG_CORPUS" \
       ION7_GPU_LAYERS="$ION7_GPU_LAYERS" \
       ION7_CORE_SRC="$ION7_CORE_SRC" \
       ION7_LLM_SRC="$ION7_LLM_SRC" \
       ION7_GRAMMAR_SRC="$ION7_GRAMMAR_SRC" \
       ION7_LIBLLAMA_PATH="$ION7_LIBLLAMA_PATH" \
       ION7_LIBGGML_PATH="$ION7_LIBGGML_PATH" \
       ION7_BRIDGE_PATH="$ION7_BRIDGE_PATH" \
       ION7_RAG_SQLITE_VEC_PATH="$ION7_RAG_SQLITE_VEC_PATH" \
       LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
       luajit "$file"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
    fi
}

for f in tests/[0-9][0-9]_*.lua; do
    [ -f "$f" ] || continue
    run_suite "$f"
done

# ── Summary ──────────────────────────────────────────────────────────────
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"
printf "  Suites: \033[32m%d passed\033[0m" "$PASS"
[ "$FAIL" -gt 0 ] && printf "  \033[31m%d FAILED\033[0m" "$FAIL"
[ "$SKIP" -gt 0 ] && printf "  \033[33m%d skipped\033[0m" "$SKIP"
printf "\n\033[1m%s\033[0m\n" "$(printf '═%.0s' {1..60})"

[ "$FAIL" -eq 0 ]
