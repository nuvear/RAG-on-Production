# Session 2 references and implementation boundaries

Reviewed 9 September 2026. Classroom code and synthetic data are original. Source chapter concepts were selected from the user's corrected *Hands-On RAG for Production* vault by Mendelevitch & Bao; the book text and full chapter implementations are not redistributed here.

| Workshop lab | Original chapter | Selected concepts | Classroom adaptation |
|---|---|---|---|
| 5: RAG platform | 5: The RAG Platform | Managed ingestion, retrieval API, attributes, lifecycle | OpenAI Vector Stores plus explicit Responses generation; not Vectara feature parity |
| 6: AI agents | 7: From RAG to AI Agents | Tool calling, execution loop, traces, guardrails | Two local read-only tools; no framework, MCP server or external write action |
| 7: Multimodal RAG | 8: Multimodal RAG | Retrieval versus visual reading, source identity | Text-embedded captions select original PNG charts; pixels reach the vision model |
| 8: Knowledge graphs | 9: Knowledge-Enhanced RAG | Typed relations, multi-hop constraints, hybrid retrieval | Curated in-memory graph with current/verified edges; no Neo4j or Microsoft GraphRAG deployment |

Official API references for the implemented request fields:

- [Retrieval](https://developers.openai.com/api/docs/guides/retrieval): managed vector stores and retrieval concepts.
- [Create vector store](https://developers.openai.com/api/reference/resources/vector_stores/methods/create): store creation and expiry fields.
- [Attach a file](https://developers.openai.com/api/reference/resources/vector_stores/subresources/files/methods/create): indexing association and file attributes.
- [Search vector store](https://developers.openai.com/api/reference/resources/vector_stores/methods/search): search request, attribute filter and returned passages.
- [Delete file](https://developers.openai.com/api/reference/resources/files/methods/delete): deleting original uploaded file objects separately from store cleanup.
- [Function calling](https://developers.openai.com/api/docs/guides/function-calling): tool definitions, argument schema and `function_call_output` with `call_id`.
- [Images and vision](https://developers.openai.com/api/docs/guides/images-vision): image inputs, data URLs and detail settings.
- [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini): configured model capabilities; access depends on the student's project.
- [Hosted file search](https://developers.openai.com/api/docs/guides/tools-file-search): follow-on comparison only. This workshop calls the Vector Stores search endpoint explicitly rather than asking the model to invoke the hosted `file_search` tool.
- [Pricing](https://developers.openai.com/api/docs/pricing): check current generation, embeddings and vector storage rates before class. No fixed per-student bill is promised.

The relationship model and its synthetic test cases are classroom design decisions. `verified=True` represents already-curated evidence; it does not prove authenticity. No claim is made that current metadata is equivalent to server-side authorization, that citations prove entailment, or that one vision/agent test measures production reliability.
