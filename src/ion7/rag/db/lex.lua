--- @module ion7.rag.db.lex
--- @author  ion7 / Ion7 Project Contributors
---
--- FTS5 BM25 lexical tier (`idx.chunks_fts`).
---
--- Each row stores the chunk text under a `rowid` that matches
--- `chunks.id`. Search uses FTS5's built-in BM25 ranker. Distance
--- values are reported where **smaller is better**, mirroring the
--- convention of the vector tier so fusion can treat both index
--- sources uniformly :
---
---   - vec hits  : Hamming or L2 distance ; 0 = perfect match.
---   - lex hits  : raw `bm25(chunks_fts)` value ; negative for matches,
---                 more-negative = better.
---
--- The schema sets `unicode61 remove_diacritics 2` as the tokeniser,
--- which handles Latin-script multilingual content cleanly. CJK falls
--- back to per-character splits ; corpora dominated by CJK / Arabic /
--- Hebrew should use the `trigram` tokeniser or a pluggable ICU one.

local M = {}

-- ── Insert ──────────────────────────────────────────────────────────────

--- Insert or replace the FTS row for `chunk_id`. The rowid in
--- `idx.chunks_fts` must match `chunks.id` so retrieval can join
--- across tiers without a secondary lookup.
---
--- @param  h         ion7.rag.db.Handle
--- @param  chunk_id  integer
--- @param  text      string  The text to index ; supplied verbatim, not
---                    parsed for FTS5 query syntax.
--- @raise When the INSERT fails.
function M.upsert(h, chunk_id, text)
    local del = h:prepare("DELETE FROM idx.chunks_fts WHERE rowid = ?")
    del:bind_values(chunk_id)
    del:step()
    del:finalize()

    local ins = h:prepare([[
        INSERT INTO idx.chunks_fts(rowid, text) VALUES (?, ?)
    ]])
    ins:bind_values(chunk_id, text)
    local rc = ins:step()
    ins:finalize()
    if rc ~= 101 then -- SQLITE_DONE
        error("lex.upsert : step failed (rc=" .. tostring(rc) ..
              ", " .. tostring(h:conn():errmsg()) .. ")", 2)
    end
end

--- Bulk-upsert FTS rows. Single SQLite transaction, two reused
--- prepared statements.
---
--- @param  h     ion7.rag.db.Handle
--- @param  rows  table[]  `{ { chunk_id = integer, text = string }, ... }`
--- @raise When any row is missing a required field or an INSERT fails.
function M.upsert_many(h, rows)
    h:transaction(function()
        local del = h:prepare("DELETE FROM idx.chunks_fts WHERE rowid = ?")
        local ins = h:prepare("INSERT INTO idx.chunks_fts(rowid, text) VALUES (?, ?)")
        for i, row in ipairs(rows) do
            assert(row.chunk_id and row.text,
                   "lex.upsert_many[" .. i .. "] : missing chunk_id / text")

            del:bind_values(row.chunk_id) ; del:step() ; del:reset()
            ins:bind_values(row.chunk_id, row.text)
            local rc = ins:step()
            if rc ~= 101 then
                ins:finalize() ; del:finalize()
                error("lex.upsert_many[" .. i .. "] : rc=" .. tostring(rc), 2)
            end
            ins:reset()
        end
        ins:finalize()
        del:finalize()
    end)
end

--- Delete one FTS row by rowid (= chunk_id).
---
--- @param  h         ion7.rag.db.Handle
--- @param  chunk_id  integer
--- @return integer  Number of rows the DELETE affected (0 or 1).
function M.delete(h, chunk_id)
    local stmt = h:prepare("DELETE FROM idx.chunks_fts WHERE rowid = ?")
    stmt:bind_values(chunk_id)
    stmt:step()
    stmt:finalize()
    return h:changes()
end

--- Total number of rows currently in `idx.chunks_fts`.
--- @param  h  ion7.rag.db.Handle
--- @return integer
function M.count(h)
    local stmt = h:prepare("SELECT COUNT(*) FROM idx.chunks_fts")
    stmt:step()
    local n = stmt:get_value(0)
    stmt:finalize()
    return n
end

-- ── Search ──────────────────────────────────────────────────────────────

--- BM25 search. The query string is passed verbatim to FTS5's `MATCH`
--- operator, so callers can use the full FTS5 query language (phrase
--- queries, prefix matches with `pat*`, `NEAR()`, boolean combinations).
--- Sanitising end-user input against FTS5 syntax is the caller's
--- responsibility — see
--- https://www.sqlite.org/fts5.html#full_text_query_syntax.
---
--- @param  h     ion7.rag.db.Handle
--- @param  query string   FTS5 query expression.
--- @param  k     integer  Maximum number of hits to return (`k > 0`).
--- @return Hit[]          `{ { chunk_id, distance }, ... }` ordered by
---                  ascending distance. `distance` is the raw
---                  `bm25(chunks_fts)` value (negative for matches,
---                  more-negative = stronger match).
--- @raise When the query is empty or `k <= 0`.
function M.search(h, query, k)
    assert(query and query ~= "", "lex.search : query is required")
    assert(k and k > 0, "lex.search : k must be > 0")

    local stmt = h:prepare([[
        SELECT rowid, bm25(chunks_fts) AS distance
        FROM idx.chunks_fts
        WHERE chunks_fts MATCH ?
        ORDER BY distance
        LIMIT ?
    ]])
    stmt:bind_values(query, k)

    local hits = {}
    while stmt:step() == 100 do -- SQLITE_ROW
        hits[#hits + 1] = {
            chunk_id = stmt:get_value(0),
            distance = stmt:get_value(1),
        }
    end
    stmt:finalize()
    return hits
end

return M
