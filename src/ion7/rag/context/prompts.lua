--- @module ion7.rag.context.prompts
--- @author  ion7 / Ion7 Project Contributors
---
--- Default prompts for the Contextual Retrieval enricher (Anthropic,
--- Sept 2024 ; formalised in Pham et al., arXiv:2504.19754, ECIR 2025).
---
--- The contextualiser produces one to three concise sentences of
--- situational context for an excerpt within a document, so that BM25
--- and dense retrieval can recover an isolated chunk's anchor (which
--- entity, which section, which date) without the surrounding pages.
---
--- Two strings are exposed and overridable at `Enricher.new` :
---
---   SYSTEM       Short instruction-style system prompt.
---   USER_FORMAT  Format string with two `%s` placeholders : full
---                document text first, chunk text second.
---
--- The system prompt is intentionally language-neutral ; the
--- generated context comes out in whatever language the model reads
--- in the document.

local M = {}

M.SYSTEM = [[You are a precise contextualization assistant. Given a document and an excerpt taken from it, produce one to three concise sentences that situate the excerpt within the wider document : which section it belongs to, which entity / topic / date it concerns, what came just before. Reply with the situating context only — no preamble, no quoting of the excerpt, no headings.]]

M.USER_FORMAT = [[<document>
%s
</document>

Here is an excerpt from that document :
<excerpt>
%s
</excerpt>

Provide the short situating context now.]]

return M
