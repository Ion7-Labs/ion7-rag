--- @module ion7.rag.util.log
--- @author  ion7 / Ion7 Project Contributors
---
--- Minimal level-filtered logger. Silent by default ; once the level
--- is raised, emits `[ion7.rag] LEVEL : message` to stderr.
---
--- Level scale (matches `ion7.core.util.log`) :
---
---     0  silent
---     1  error
---     2  warn
---     3  info
---     4  debug
---
--- At module load, the current level of `ion7.core.util.log` is read
--- once and adopted as the initial level, so a single
--- `ion7.init({ log_level = 3 })` call propagates through both layers.
---
--- @usage
---   local log = require "ion7.rag.util.log"
---   log.set_level(2)
---   log.warn("ingestion lagging — context Pool saturated")

local M = { level = 0, _stream = io.stderr }

local LABELS = { [1] = "ERROR", [2] = "WARN", [3] = "INFO", [4] = "DEBUG" }

--- Write a single line at `level_int` if it passes the current filter.
local function emit(level_int, msg)
    if level_int > M.level then return end
    M._stream:write(string.format("[ion7.rag] %s : %s\n",
        LABELS[level_int] or "?", msg))
    M._stream:flush()
end

--- Set the verbosity threshold. Out-of-range values fall back to 0.
--- @param  n  integer  0–4.
function M.set_level(n)
    M.level = (type(n) == "number" and n >= 0 and n <= 4) and n or 0
end

--- Redirect output to a writable stream.
--- @param  stream  table  Any object exposing `write(s)` and `flush()`
---                  (e.g. `io.stdout`, an open file handle).
function M.set_stream(stream)
    if stream and type(stream.write) == "function" then
        M._stream = stream
    end
end

--- Emit at level 1 (error) when the current level >= 1.
--- @param  msg  string
function M.error(msg) emit(1, msg) end

--- Emit at level 2 (warn) when the current level >= 2.
--- @param  msg  string
function M.warn (msg) emit(2, msg) end

--- Emit at level 3 (info) when the current level >= 3.
--- @param  msg  string
function M.info (msg) emit(3, msg) end

--- Emit at level 4 (debug) when the current level >= 4.
--- @param  msg  string
function M.debug(msg) emit(4, msg) end

-- Adopt ion7.core.util.log's current level on first require, when it's
-- reachable. Failures here are silent — this module must work without
-- ion7-core on the package path.
do
    local ok, core_log = pcall(require, "ion7.core.util.log")
    if ok and core_log and type(core_log.snapshot) == "function" then
        local snap = core_log.snapshot()
        if snap and type(snap.level) == "number" then
            M.level = snap.level
        end
    end
end

return M
