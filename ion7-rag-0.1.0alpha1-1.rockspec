package = "ion7-rag"
version = "0.1.0alpha1-1"

source = {
    url = "git+https://github.com/Ion7-Labs/ion7-rag.git",
    tag = "v0.1.0-alpha1",
}

description = {
    summary  = "Local-first RAG library on top of ion7-core / ion7-llm / ion7-grammar",
    detailed = [[
        ion7-rag — Retrieval-Augmented Generation as a pure-Lua library.

        Built on the ion7 stack :
          - ion7-core for embeddings, rerankers, generation.
          - ion7-llm for chat pipeline, multi-session pool, RadixAttention.
          - ion7-grammar for constrained-output JSON / listwise / reflection.

        Storage substrate : sqlite-vec (dense vectors) + SQLite FTS5 (BM25)
        across two SQLite files — chunks.db (canonical text + metadata) and
        index.db (disposable vectors + lexical index + HyPE aux tier).

        Surface : recursive and late chunkers, Anthropic Contextual
        Retrieval, hybrid retrieval with RRF / DBSF / CC fusion,
        pointwise yes/no reranker, HyPE Option B, TF-IDF query router,
        CRAG and Self-RAG agents, RAGAs-style evaluation
        (Faithfulness, ContextPrecision, Lynx).

        Library, not a server. No HTTP, no SSE, no CLI binary.
    ]],
    homepage = "https://github.com/Ion7-Labs/ion7-rag",
    license  = "MIT",
}

dependencies = {
    "lua >= 5.1",
    "ion7-core >= 0.1.0beta4",
    "ion7-llm >= 0.2.0beta1",
    "ion7-grammar >= 0.2.0beta1",
    "lsqlite3 >= 0.9.5",
    "lpeg >= 1.0",
    "cmark >= 0.30",
    "gumbo >= 0.5",
    -- sqlite-vec is loaded at runtime through SQLite's extension loader
    -- (vec0 virtual table) ; lsqlite3 does not declare it as a Lua dep.
}

build = {
    type    = "builtin",
    modules = {
        ["ion7.rag"]                 = "src/ion7/rag/init.lua",
        ["ion7.rag.db"]              = "src/ion7/rag/db/init.lua",
        ["ion7.rag.db.schema"]       = "src/ion7/rag/db/schema.lua",
        ["ion7.rag.db.chunks"]       = "src/ion7/rag/db/chunks.lua",
        ["ion7.rag.db.vec"]          = "src/ion7/rag/db/vec.lua",
        ["ion7.rag.db.lex"]          = "src/ion7/rag/db/lex.lua",
        ["ion7.rag.db.hype"]         = "src/ion7/rag/db/hype.lua",
        ["ion7.rag.loader"]          = "src/ion7/rag/loader/init.lua",
        ["ion7.rag.loader.text"]     = "src/ion7/rag/loader/text.lua",
        ["ion7.rag.loader.markdown"] = "src/ion7/rag/loader/markdown.lua",
        ["ion7.rag.loader.html"]     = "src/ion7/rag/loader/html.lua",
        ["ion7.rag.chunk"]           = "src/ion7/rag/chunk/init.lua",
        ["ion7.rag.chunk.recursive"] = "src/ion7/rag/chunk/recursive.lua",
        ["ion7.rag.chunk.late"]      = "src/ion7/rag/chunk/late.lua",
        ["ion7.rag.fusion"]          = "src/ion7/rag/fusion/init.lua",
        ["ion7.rag.fusion.rrf"]      = "src/ion7/rag/fusion/rrf.lua",
        ["ion7.rag.fusion.dbsf"]     = "src/ion7/rag/fusion/dbsf.lua",
        ["ion7.rag.fusion.cc"]       = "src/ion7/rag/fusion/cc.lua",
        ["ion7.rag.retrieve"]        = "src/ion7/rag/retrieve.lua",
        ["ion7.rag.embed"]           = "src/ion7/rag/embed.lua",
        ["ion7.rag.rerank"]          = "src/ion7/rag/rerank/init.lua",
        ["ion7.rag.rerank.pointwise"]= "src/ion7/rag/rerank/pointwise.lua",
        ["ion7.rag.context"]         = "src/ion7/rag/context/init.lua",
        ["ion7.rag.context.prompts"] = "src/ion7/rag/context/prompts.lua",
        ["ion7.rag.pipeline"]        = "src/ion7/rag/pipeline.lua",
        ["ion7.rag.route"]           = "src/ion7/rag/route/init.lua",
        ["ion7.rag.route.tfidf"]     = "src/ion7/rag/route/tfidf.lua",
        ["ion7.rag.hype"]            = "src/ion7/rag/hype.lua",
        ["ion7.rag.agent"]           = "src/ion7/rag/agent/init.lua",
        ["ion7.rag.agent.prompts"]   = "src/ion7/rag/agent/prompts.lua",
        ["ion7.rag.agent.crag"]      = "src/ion7/rag/agent/crag.lua",
        ["ion7.rag.agent.self_rag"]  = "src/ion7/rag/agent/self_rag.lua",
        ["ion7.rag.eval"]                 = "src/ion7/rag/eval/init.lua",
        ["ion7.rag.eval.prompts"]         = "src/ion7/rag/eval/prompts.lua",
        ["ion7.rag.eval.faithfulness"]    = "src/ion7/rag/eval/faithfulness.lua",
        ["ion7.rag.eval.context_precision"] = "src/ion7/rag/eval/context_precision.lua",
        ["ion7.rag.eval.lynx"]            = "src/ion7/rag/eval/lynx.lua",
        ["ion7.rag.util.log"]        = "src/ion7/rag/util/log.lua",
    },
}
