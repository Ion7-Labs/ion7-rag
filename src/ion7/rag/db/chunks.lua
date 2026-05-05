--- @module ion7.rag.db.chunks
--- @author  ion7 / Ion7 Project Contributors
---
--- Docs + chunks CRUD on the truth tier (`chunks.db`, schema `main`).
---
--- Doc rows carry the canonical document identity ; chunk rows carry
--- the canonical text and provenance (doc reference + char span).
--- Every row is identified by an autoincrementing INTEGER primary key
--- — the same value is used as `chunk_id` in `idx.chunks_vec`,
--- `idx.hype_vec.chunk_id`, and as `rowid` in `idx.chunks_fts`, so
--- cross-tier joins never need a secondary lookup.
---
--- Every public function takes a `Handle` (from `ion7.rag.db.open`) as
--- its first argument. None of them touch the index tier — that lives
--- in `ion7.rag.db.vec`, `ion7.rag.db.lex`, and `ion7.rag.db.hype`.

local M = {}

-- ── Doc operations ──────────────────────────────────────────────────────

--- Insert a doc row.
---
--- @param  h    ion7.rag.db.Handle
--- @param  doc  table {
---     doc_id      string   Stable caller-provided identifier ; UNIQUE.
---     format      string   "text" | "markdown" | "html" | ...
---     source_uri  string?  Path / URL / nil for in-memory documents.
---     title       string?
---     ingested_at integer? Unix epoch. Defaults to `os.time()`.
---     meta_json   string?  Format-specific blob ; caller serialises.
--- }
--- @return integer  The new doc primary key.
--- @raise When `doc_id` or `format` is missing, or when the underlying
---        INSERT fails (e.g. UNIQUE collision on `doc_id`).
function M.insert_doc(h, doc)
    assert(doc.doc_id, "insert_doc : doc.doc_id is required")
    assert(doc.format, "insert_doc : doc.format is required")

    local stmt = h:prepare([[
        INSERT INTO docs(doc_id, source_uri, title, format, ingested_at, meta_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ]])
    stmt:bind_values(
        doc.doc_id,
        doc.source_uri,
        doc.title,
        doc.format,
        doc.ingested_at or os.time(),
        doc.meta_json
    )
    local rc = stmt:step()
    stmt:finalize()
    if rc ~= 101 then -- SQLITE_DONE
        error("insert_doc : step failed (rc=" .. tostring(rc) ..
              ", " .. tostring(h:conn():errmsg()) .. ")", 2)
    end
    return h:last_insert_rowid()
end

--- Look up a doc row by its caller-provided `doc_id`.
---
--- @param  h       ion7.rag.db.Handle
--- @param  doc_id  string
--- @return table?  Full row, or `nil` when the id is unknown.
function M.get_doc_by_id(h, doc_id)
    local stmt = h:prepare([[
        SELECT id, doc_id, source_uri, title, format, ingested_at, meta_json
        FROM docs WHERE doc_id = ?
    ]])
    stmt:bind_values(doc_id)
    local row
    if stmt:step() == 100 then -- SQLITE_ROW
        row = {
            id          = stmt:get_value(0),
            doc_id      = stmt:get_value(1),
            source_uri  = stmt:get_value(2),
            title       = stmt:get_value(3),
            format      = stmt:get_value(4),
            ingested_at = stmt:get_value(5),
            meta_json   = stmt:get_value(6),
        }
    end
    stmt:finalize()
    return row
end

--- Look up a doc row by its internal primary key.
---
--- @param  h       ion7.rag.db.Handle
--- @param  doc_pk  integer
--- @return table?  Full row, or `nil` when the pk is unknown.
function M.get_doc_by_pk(h, doc_pk)
    local stmt = h:prepare([[
        SELECT id, doc_id, source_uri, title, format, ingested_at, meta_json
        FROM docs WHERE id = ?
    ]])
    stmt:bind_values(doc_pk)
    local row
    if stmt:step() == 100 then
        row = {
            id          = stmt:get_value(0),
            doc_id      = stmt:get_value(1),
            source_uri  = stmt:get_value(2),
            title       = stmt:get_value(3),
            format      = stmt:get_value(4),
            ingested_at = stmt:get_value(5),
            meta_json   = stmt:get_value(6),
        }
    end
    stmt:finalize()
    return row
end

--- Delete a doc and (via `ON DELETE CASCADE`) all its chunk rows.
--- Index-tier rows for those chunks are NOT cascade-deleted ;
--- the caller is responsible for `db.vec.delete`, `db.lex.delete`,
--- and `db.hype.delete_for_chunk` if the index needs to stay clean.
---
--- @param  h       ion7.rag.db.Handle
--- @param  doc_pk  integer
--- @return integer  Number of rows the DELETE affected.
function M.delete_doc(h, doc_pk)
    local stmt = h:prepare("DELETE FROM docs WHERE id = ?")
    stmt:bind_values(doc_pk)
    stmt:step()
    stmt:finalize()
    return h:changes()
end

--- Iterate all doc rows, most-recently-ingested first.
---
--- @param  h  ion7.rag.db.Handle
--- @return function  Iterator suitable for `for row in iter_docs(h) do`.
function M.iter_docs(h)
    return h:conn():nrows([[
        SELECT id, doc_id, source_uri, title, format, ingested_at, meta_json
        FROM docs ORDER BY ingested_at DESC
    ]])
end

-- ── Chunk operations ────────────────────────────────────────────────────

--- Insert a single chunk row.
---
--- @param  h      ion7.rag.db.Handle
--- @param  chunk  table {
---     doc_pk          integer   Foreign key into `docs.id`.
---     section         string?   E.g. "Header > Body" for markdown / HTML.
---     char_start      integer   Inclusive byte offset in the source doc.
---     char_end        integer   Exclusive byte offset.
---     n_tokens        integer?  Optional token count from the chunker.
---     raw_text        string    The chunk text verbatim.
---     contextual_text string?   Anthropic-style prepended context, when
---                      an enricher is wired into the ingest path.
---     meta_json       string?   Caller-serialised metadata blob.
--- }
--- @return integer  The new chunk primary key.
--- @raise When any required field is missing or the INSERT fails.
function M.insert_chunk(h, chunk)
    assert(chunk.doc_pk,     "insert_chunk : chunk.doc_pk is required")
    assert(chunk.char_start, "insert_chunk : chunk.char_start is required")
    assert(chunk.char_end,   "insert_chunk : chunk.char_end is required")
    assert(chunk.raw_text,   "insert_chunk : chunk.raw_text is required")

    local stmt = h:prepare([[
        INSERT INTO chunks(
            doc_pk, section, char_start, char_end,
            n_tokens, raw_text, contextual_text, meta_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ]])
    stmt:bind_values(
        chunk.doc_pk,
        chunk.section,
        chunk.char_start,
        chunk.char_end,
        chunk.n_tokens,
        chunk.raw_text,
        chunk.contextual_text,
        chunk.meta_json
    )
    local rc = stmt:step()
    stmt:finalize()
    if rc ~= 101 then
        error("insert_chunk : step failed (rc=" .. tostring(rc) ..
              ", " .. tostring(h:conn():errmsg()) .. ")", 2)
    end
    return h:last_insert_rowid()
end

--- Bulk-insert a batch of chunk rows. Wraps a single SQLite
--- transaction and reuses one prepared statement.
---
--- @param  h             ion7.rag.db.Handle
--- @param  chunks_array  table[]  Rows shaped like `insert_chunk`'s `chunk`.
--- @return integer[]              The new chunk primary keys, in input order.
--- @raise When any row misses a required field, or when an INSERT fails.
function M.insert_chunks(h, chunks_array)
    local pks = {}
    h:transaction(function()
        local stmt = h:prepare([[
            INSERT INTO chunks(
                doc_pk, section, char_start, char_end,
                n_tokens, raw_text, contextual_text, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ]])
        for i, chunk in ipairs(chunks_array) do
            assert(chunk.doc_pk and chunk.char_start and chunk.char_end
                   and chunk.raw_text,
                   "insert_chunks[" .. i .. "] : missing required field")
            stmt:bind_values(
                chunk.doc_pk,
                chunk.section,
                chunk.char_start,
                chunk.char_end,
                chunk.n_tokens,
                chunk.raw_text,
                chunk.contextual_text,
                chunk.meta_json
            )
            local rc = stmt:step()
            if rc ~= 101 then
                stmt:finalize()
                error("insert_chunks[" .. i .. "] : rc=" .. tostring(rc), 2)
            end
            pks[i] = h:last_insert_rowid()
            stmt:reset()
        end
        stmt:finalize()
    end)
    return pks
end

--- Look up a chunk row by its primary key.
---
--- @param  h         ion7.rag.db.Handle
--- @param  chunk_pk  integer
--- @return table?    Full row, or `nil` when the pk is unknown.
function M.get_chunk(h, chunk_pk)
    local stmt = h:prepare([[
        SELECT id, doc_pk, section, char_start, char_end,
               n_tokens, raw_text, contextual_text, meta_json
        FROM chunks WHERE id = ?
    ]])
    stmt:bind_values(chunk_pk)
    local row
    if stmt:step() == 100 then
        row = {
            id              = stmt:get_value(0),
            doc_pk          = stmt:get_value(1),
            section         = stmt:get_value(2),
            char_start      = stmt:get_value(3),
            char_end        = stmt:get_value(4),
            n_tokens        = stmt:get_value(5),
            raw_text        = stmt:get_value(6),
            contextual_text = stmt:get_value(7),
            meta_json       = stmt:get_value(8),
        }
    end
    stmt:finalize()
    return row
end

--- Fetch many chunk rows by their primary keys, preserving input order.
--- Used for hydrating retrieval hits in one round-trip.
---
--- @param  h    ion7.rag.db.Handle
--- @param  pks  integer[]  Chunk primary keys.
--- @return table[]          Chunk rows in the same order as `pks`.
function M.get_chunks(h, pks)
    if #pks == 0 then return {} end

    -- Expand placeholders ourselves : lsqlite3 binds varargs but `or`
    -- truncates multi-return values, so passing `unpack(pks)` would
    -- silently bind only the first one. Per-index bind is unambiguous.
    local placeholders = {}
    for i = 1, #pks do placeholders[i] = "?" end
    local sql = string.format([[
        SELECT id, doc_pk, section, char_start, char_end,
               n_tokens, raw_text, contextual_text, meta_json
        FROM chunks WHERE id IN (%s)
    ]], table.concat(placeholders, ","))

    local stmt = h:prepare(sql)
    for i, pk in ipairs(pks) do stmt:bind(i, pk) end

    local by_id = {}
    while stmt:step() == 100 do
        local id = stmt:get_value(0)
        by_id[id] = {
            id              = id,
            doc_pk          = stmt:get_value(1),
            section         = stmt:get_value(2),
            char_start      = stmt:get_value(3),
            char_end        = stmt:get_value(4),
            n_tokens        = stmt:get_value(5),
            raw_text        = stmt:get_value(6),
            contextual_text = stmt:get_value(7),
            meta_json       = stmt:get_value(8),
        }
    end
    stmt:finalize()

    local out = {}
    for i, pk in ipairs(pks) do out[i] = by_id[pk] end
    return out
end

--- Iterate every chunk row that belongs to `doc_pk`, ordered by
--- ascending `char_start`.
---
--- @param  h       ion7.rag.db.Handle
--- @param  doc_pk  integer
--- @return function  Stateful iterator yielding one chunk row per call,
---                    nil when exhausted. The underlying statement is
---                    finalised on exhaustion.
function M.iter_chunks_for_doc(h, doc_pk)
    local stmt = h:prepare([[
        SELECT id, doc_pk, section, char_start, char_end,
               n_tokens, raw_text, contextual_text, meta_json
        FROM chunks WHERE doc_pk = ? ORDER BY char_start ASC
    ]])
    stmt:bind_values(doc_pk)
    return function()
        if stmt:step() == 100 then
            return {
                id              = stmt:get_value(0),
                doc_pk          = stmt:get_value(1),
                section         = stmt:get_value(2),
                char_start      = stmt:get_value(3),
                char_end        = stmt:get_value(4),
                n_tokens        = stmt:get_value(5),
                raw_text        = stmt:get_value(6),
                contextual_text = stmt:get_value(7),
                meta_json       = stmt:get_value(8),
            }
        end
        stmt:finalize()
        return nil
    end
end

--- Count chunks across the whole `chunks` table, or restricted to one doc.
---
--- @param  h       ion7.rag.db.Handle
--- @param  doc_pk  integer?  When set, count only chunks of that doc.
--- @return integer
function M.count_chunks(h, doc_pk)
    local stmt
    if doc_pk then
        stmt = h:prepare("SELECT COUNT(*) FROM chunks WHERE doc_pk = ?")
        stmt:bind_values(doc_pk)
    else
        stmt = h:prepare("SELECT COUNT(*) FROM chunks")
    end
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n
end

return M
