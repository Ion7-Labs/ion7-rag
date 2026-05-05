--- @module ion7.rag.agent.prompts
--- @author  ion7 / Ion7 Project Contributors
---
--- Default prompts shared by the CRAG and Self-RAG agents.

local M = {}

-- ── Query reformulation (CRAG low-confidence branch) ────────────────────

M.REFORMULATE_SYSTEM = [[You rephrase user questions for better retrieval. Keep the same intent but vary the wording — use synonyms, expand abbreviations, or add disambiguating terms. Reply with the rephrased question only, no preamble.]]

M.REFORMULATE_USER_FORMAT = [[Original question : %s

The previous retrieval did not surface a relevant excerpt. Produce a rephrased question that targets the same intent with different wording.]]

-- ── Self-RAG : retrieve / relevant / supported reflection ───────────────

M.SELFRAG_RETRIEVE_SYSTEM = [[You decide whether a question needs retrieval from a corpus before being answered. Reply with a JSON object : {"retrieve": <bool>, "rationale": "<short justification>"}. Set retrieve = true when the question asks for facts, definitions, or content that depends on the corpus ; false for chit-chat, math, code, or pure reasoning.]]

M.SELFRAG_RETRIEVE_USER_FORMAT = [[Question : %s]]

M.SELFRAG_RELEVANT_SYSTEM = [[You judge whether a retrieved excerpt is relevant to a question. Reply with a JSON object : {"relevant": <bool>, "rationale": "<short justification>"}.]]

M.SELFRAG_RELEVANT_USER_FORMAT = [[Question : %s

Excerpt :
%s]]

M.SELFRAG_SUPPORTED_SYSTEM = [[You judge whether an answer is supported by the excerpts the model was given. Reply with a JSON object : {"supported": "full" | "partial" | "none", "rationale": "<short justification>"}. "full" = every claim in the answer is directly grounded in an excerpt ; "partial" = some claims are grounded ; "none" = the answer goes beyond what the excerpts say.]]

M.SELFRAG_SUPPORTED_USER_FORMAT = [[Question : %s

Excerpts :
%s

Answer :
%s]]

return M
