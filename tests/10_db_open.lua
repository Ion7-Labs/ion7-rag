#!/usr/bin/env luajit
--- @module tests.10_db_open
--- @author  ion7 / Ion7 Project Contributors
---
--- Verify the Db handle : open, ATTACH index, sqlite-vec extension load,
--- schema bootstrap, dimension-stamp validation across re-opens.

local T = require "tests.framework"
local H = require "tests.helpers"

T.suite("ion7.rag.db.open — open / attach / bootstrap")

H.require_lsqlite3(T)
local sqlite_vec_path = H.require_sqlite_vec(T)

local db = require "ion7.rag.db"

-- Clean state for each test : open / close, then nuke files.
local function with_fresh_pair(fn)
    local cp, ip = H.tmp_db_pair("10")
    local ok, err = pcall(fn, cp, ip)
    H.try_remove(cp) ; H.try_remove(ip)
    H.try_remove(cp .. "-wal") ; H.try_remove(cp .. "-shm")
    H.try_remove(ip .. "-wal") ; H.try_remove(ip .. "-shm")
    if not ok then error(err, 0) end
end

T.test("open() creates both files and bootstraps schema", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path     = cp,
            index_path      = ip,
            embed_dim       = 1024,
            binary_dim      = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        T.is_type(h, "table")
        T.eq(h:embed_dim(),  1024)
        T.eq(h:binary_dim(), 192)
        h:close()

        -- Both files now exist on disk.
        local fc = io.open(cp, "rb") ; T.ok(fc, "chunks.db not created") ; fc:close()
        local fi = io.open(ip, "rb") ; T.ok(fi, "index.db not created")  ; fi:close()
    end)
end)

T.test("open() refuses missing required opts", function()
    with_fresh_pair(function(cp, ip)
        T.err(function()
            db.open({ index_path = ip, embed_dim = 1024, binary_dim = 192,
                      sqlite_vec_path = sqlite_vec_path })
        end, "chunks_path")
        T.err(function()
            db.open({ chunks_path = cp, embed_dim = 1024, binary_dim = 192,
                      sqlite_vec_path = sqlite_vec_path })
        end, "index_path")
        T.err(function()
            db.open({ chunks_path = cp, index_path = ip, binary_dim = 192,
                      sqlite_vec_path = sqlite_vec_path })
        end, "embed_dim")
        T.err(function()
            db.open({ chunks_path = cp, index_path = ip, embed_dim = 1024,
                      sqlite_vec_path = sqlite_vec_path })
        end, "binary_dim")
    end)
end)

T.test("close() is idempotent", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        h:close()
        T.no_error(function() h:close() end, "second close raised")
    end)
end)

T.test("re-open with same dims succeeds", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        h:close()

        local h2 = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        T.eq(h2:embed_dim(),  1024)
        T.eq(h2:binary_dim(), 192)
        h2:close()
    end)
end)

T.test("re-open with mismatched embed_dim errors clearly", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        h:close()

        T.err(function()
            db.open({
                chunks_path = cp, index_path = ip,
                embed_dim = 768, binary_dim = 192,
                sqlite_vec_path = sqlite_vec_path,
            })
        end, "embed_dim mismatch")
    end)
end)

T.test("re-open with mismatched binary_dim errors clearly", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        h:close()

        T.err(function()
            db.open({
                chunks_path = cp, index_path = ip,
                embed_dim = 1024, binary_dim = 256,
                sqlite_vec_path = sqlite_vec_path,
            })
        end, "binary_dim mismatch")
    end)
end)

T.test("vec_version() resolves through the loaded extension", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        local v
        for r in h:conn():nrows("SELECT vec_version() AS v") do v = r.v end
        T.is_type(v, "string")
        T.gt(#v, 0)
        h:close()
    end)
end)

T.test("FTS5 is available (CREATE VIRTUAL TABLE succeeded at bootstrap)", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })
        local n
        for r in h:conn():nrows(
            "SELECT COUNT(*) AS n FROM idx.sqlite_master WHERE name = 'chunks_fts'")
        do n = r.n end
        T.eq(n, 1, "chunks_fts virtual table missing")
        h:close()
    end)
end)

T.test("transaction rolls back on error", function()
    with_fresh_pair(function(cp, ip)
        local h = db.open({
            chunks_path = cp, index_path = ip,
            embed_dim = 1024, binary_dim = 192,
            sqlite_vec_path = sqlite_vec_path,
        })

        T.err(function()
            h:transaction(function()
                h:exec([[INSERT INTO docs(doc_id, format, ingested_at)
                         VALUES('roll-back-me', 'text', 0)]])
                error("boom")
            end)
        end, "boom")

        local n
        for r in h:conn():nrows(
            "SELECT COUNT(*) AS n FROM docs WHERE doc_id = 'roll-back-me'")
        do n = r.n end
        T.eq(n, 0, "row was not rolled back")
        h:close()
    end)
end)

local ok = T.summary()
os.exit(ok and 0 or 1)
