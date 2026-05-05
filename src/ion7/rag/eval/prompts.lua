--- @module ion7.rag.eval.prompts
--- @author  ion7 / Ion7 Project Contributors
---
--- Shared judge prompts for the RAGAs-style evaluation metrics. Every
--- "yes / no" judge re-uses `ion7.rag.rerank.Pointwise` under the hood
--- — the system prompt is what specialises the call.

local M = {}

-- ── Faithfulness ────────────────────────────────────────────────────────

M.CLAIM_EXTRACT_SYSTEM = [[You extract atomic factual claims from a passage. A claim is one verifiable proposition (a fact, a date, a quantity, a relation). Reply with one claim per line, prefixed with "- ". Skip claims that are pure opinion or unverifiable. No preamble.]]

M.CLAIM_EXTRACT_USER_FORMAT = [[Passage :
%s

List the atomic factual claims, one per line.]]

M.FAITHFULNESS_JUDGE_SYSTEM = [[You decide whether a given claim is directly supported by the provided context. Answer only "yes" or "no". Treat strict equivalence — if the context does not state the claim explicitly or by a simple paraphrase, answer no.]]

M.FAITHFULNESS_JUDGE_USER_FORMAT = [[Context :
%s

Claim :
%s

Is the claim supported by the context?]]

-- ── Context Precision ──────────────────────────────────────────────────

M.CONTEXT_PRECISION_SYSTEM = [[You decide whether a given excerpt is relevant for answering a question. Answer only "yes" or "no". An excerpt is relevant when it carries information that directly contributes to the answer ; tangentially-related material is not relevant.]]

M.CONTEXT_PRECISION_USER_FORMAT = [[Question : %s

Excerpt :
%s

Is the excerpt relevant for answering the question?]]

-- ── Lynx hallucination judge ───────────────────────────────────────────
-- Lynx (Patronus AI, arXiv:2407.08488) was trained with a specific
-- prompt template ; the canonical form below mirrors the model card
-- so a real Lynx GGUF responds in the expected PASS/FAIL format.

M.LYNX_SYSTEM = [[Given the following QUESTION, DOCUMENT and ANSWER, you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information in the DOCUMENT. Output your final verdict as a single token : "PASS" if the ANSWER is faithful, or "FAIL" otherwise.]]

M.LYNX_USER_FORMAT = [[QUESTION :
%s

DOCUMENT :
%s

ANSWER :
%s]]

return M
