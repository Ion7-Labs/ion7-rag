# Installing ion7-rag

ion7-rag is a pure-Lua library on top of [`ion7-core`](https://github.com/Ion7-Labs/ion7-core),
[`ion7-llm`](https://github.com/Ion7-Labs/ion7-llm) and
[`ion7-grammar`](https://github.com/Ion7-Labs/ion7-grammar). No native code
in the ion7-rag tree itself — installation reduces to four questions :

1. **Where are ion7-core / ion7-llm / ion7-grammar ?** ion7-rag has to
   resolve their Lua sources AND the native libraries ion7-core depends
   on (`libllama.so`, `ion7_bridge.so`).
2. **Is `lsqlite3` installed and is `sqlite-vec` loadable as a SQLite
   extension ?**
3. **Where are your model files ?** Tests and examples read paths from
   `ION7_MODEL`, `ION7_EMBED_MODEL`, `ION7_RERANK_MODEL`.
4. **Where is your test corpus ?** Tests that need real documents read
   from `ION7_RAG_CORPUS`.

The remainder is path bookkeeping. Two layouts cover most setups.

---

## Layout A : sibling source checkouts (developer mode)

The layout the test suite assumes by default :

```
my-workspace/
├── ion7-core/
│   ├── src/ion7/core/...
│   ├── vendor/llama.cpp/build/bin/libllama.so
│   └── bridge/build/ion7_bridge.so
├── ion7-llm/
│   └── src/ion7/llm/...
├── ion7-grammar/
│   └── src/ion7/grammar/...
└── ion7-rag/
    ├── src/ion7/rag/...
    ├── tests/
    └── examples/
```

Workflow :

```bash
cd my-workspace/

# Build ion7-core once.
cd ion7-core
make build

# Install lsqlite3 + lpeg locally.
luarocks install --local lsqlite3
luarocks install --local lpeg

# Install sqlite-vec for the SQLite extension loader.
# See https://github.com/asg017/sqlite-vec for platform-specific steps.

# Run the ion7-rag tests.
cd ../ion7-rag
bash tests/run_all.sh
```

The test helper at [`tests/helpers.lua`](tests/helpers.lua) walks up the
directory tree and prepends `../ion7-{core,llm,grammar}/src` onto
`package.path` automatically. No env vars needed for path bootstrap when
using this layout.

---

## Layout B : luarocks

Once ion7-core / ion7-llm / ion7-grammar all ship rockspecs, the
end-user setup will be :

```bash
luarocks install --local ion7-core
luarocks install --local ion7-llm
luarocks install --local ion7-grammar
luarocks install --local ion7-rag
luarocks install --local lsqlite3 lpeg
```

Always use `--local` to scope installs to your home directory
(`~/.luarocks/`). Skip `--local` and run with `sudo` for a system-wide
install ; do not mix the two.

---

## Environment variables

Read by the test suite, the example scripts, and ion7-core's FFI loader.

| Variable               | Used by              | Purpose |
|------------------------|----------------------|---------|
| `ION7_MODEL`           | tests, examples      | Chat-tuned GGUF (CRAG loop, eval judge). |
| `ION7_EMBED_MODEL`     | tests, examples      | Embedder GGUF (default Qwen3-Embedding-0.6B). |
| `ION7_RERANK_MODEL`    | tests, examples      | Cross-encoder reranker GGUF. |
| `ION7_CONTEXT_MODEL`   | tests, examples      | Optional small contextualizer ; falls back to `ION7_MODEL`. |
| `ION7_RAG_CORPUS`      | tests, examples      | Path to a directory of test documents. |
| `ION7_GPU_LAYERS`      | tests, examples      | Override `n_gpu_layers` (default 0 = pure CPU). |
| `ION7_CORE_SRC`        | helpers.lua          | Override ion7-core source root. |
| `ION7_LLM_SRC`         | helpers.lua          | Override ion7-llm source root. |
| `ION7_GRAMMAR_SRC`     | helpers.lua          | Override ion7-grammar source root. |
| `ION7_LIBLLAMA_PATH`   | ion7-core FFI loader | Pin a specific `libllama.so`. |
| `ION7_LIBGGML_PATH`    | ion7-core FFI loader | Pin a specific `libggml.so`. |
| `ION7_BRIDGE_PATH`     | ion7-core FFI loader | Pin a specific `ion7_bridge.so`. |
| `LD_LIBRARY_PATH`      | dynamic linker       | Must include the directory holding `libllama.so.0` so `ion7_bridge.so` can resolve its rpath at load time. |
| `ION7_RAG_SQLITE_VEC_PATH` | ion7-rag             | Path to the `vec0.so` extension when not on SQLite's default extension search path. |
| `ION7_SKIP`            | tests/run_all.sh     | Whitespace-separated test files to skip. |

Per the project-wide no-hardcoded-fallbacks policy, every script that
needs a path reads it from the env, never from a built-in default.

---

## Verifying the install

The pure-Lua module load test runs without any model or corpus :

```bash
bash tests/run_all.sh
# Expect: 00_modules — every test [OK]
```

The full suite (including the model-dependent ones) runs end-to-end with :

```bash
LD_LIBRARY_PATH=$HOME/Projets/Ion7-Labs/ion7-core/vendor/llama.cpp/build/bin:$LD_LIBRARY_PATH \
ION7_LIBLLAMA_PATH=$HOME/Projets/Ion7-Labs/ion7-core/vendor/llama.cpp/build/bin/libllama.so \
ION7_LIBGGML_PATH=$HOME/Projets/Ion7-Labs/ion7-core/vendor/llama.cpp/build/bin/libggml.so \
ION7_BRIDGE_PATH=$HOME/Projets/Ion7-Labs/ion7-core/bridge/ion7_bridge.so \
ION7_MODEL=$HOME/Projets/Ion7-Labs/_models/Ministral-3-3B-Instruct-2512-UD-Q8_K_XL.gguf \
ION7_EMBED_MODEL=$HOME/Projets/Ion7-Labs/_models/Qwen3-Embedding-8B-Q8_0.gguf \
ION7_RAG_CORPUS=$HOME/Projets/Idantic/Dataset/output \
bash tests/run_all.sh
```

`LD_LIBRARY_PATH` is required because `ion7_bridge.so` declares `libllama.so.0` and `libllama-common.so.0` as runtime dependencies — without the path on the dynamic linker's search list the bridge load fails with `not found`. Set it in your shell rc once and forget it.

A capability probe from the REPL :

```bash
luajit -e '
package.path = "./src/?.lua;./src/?/init.lua;" ..
               "../ion7-core/src/?.lua;../ion7-core/src/?/init.lua;" ..
               "../ion7-llm/src/?.lua;../ion7-llm/src/?/init.lua;" ..
               "../ion7-grammar/src/?.lua;../ion7-grammar/src/?/init.lua;" ..
               package.path
local rag = require "ion7.rag"
local c = rag.capabilities()
print("ion7-rag      :", c.version)
print("ion7-core     :", c.has_core    and (c.core_version or "?")    or "missing")
print("ion7-llm      :", c.has_llm     and (c.llm_version or "?")     or "missing")
print("ion7-grammar  :", c.has_grammar and (c.grammar_version or "?") or "missing")
'
```

---

## Troubleshooting

**`module 'ion7.core' not found`** — sibling checkout layout broken.
Check that `../ion7-core/src/ion7/core/init.lua` exists, or set
`ION7_CORE_SRC` explicitly.

**`module 'ion7.llm' not found`** — same as above for ion7-llm.

**`module 'lsqlite3' not found`** —
`luarocks install --local lsqlite3`. The C build needs a working `gcc`
and `sqlite3` development headers (`sqlite-devel` on Fedora,
`libsqlite3-dev` on Debian/Ubuntu).

**`unable to load extension : sqlite-vec`** — sqlite-vec is a SQLite
extension, not a Lua rock. Build it from
[asg017/sqlite-vec](https://github.com/asg017/sqlite-vec) or grab a
pre-built shared object for your platform, then make sure the path is
discoverable by SQLite's extension loader.

**`module 'lpeg' not found`** —
`luarocks install --local lpeg`.
