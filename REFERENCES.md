# Sources and implementation references

## Curriculum source

Mendelevitch & Bao, *Hands-On RAG for Production*. Selection based on the user's corrected nine-chapter RAG Vault and its existing notebook guide. The workshop uses newly authored exercises and synthetic library data. No source book PDFs, DOCX files or chapter transcriptions are included in this repository.

## Official implementation documentation

Checked 9 September 2026:

- [Embeddings request API](https://developers.openai.com/api/reference/resources/embeddings/methods/create): batch inputs and returned vector/index fields.
- [text-embedding-3-small](https://developers.openai.com/api/docs/models/text-embedding-3-small): configured embedding model.
- [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini): configured generation model.
- [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs): strict JSON schemas, output parsing and refusal handling.
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing): check current rates before class. No fixed price is promised by this package.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html): notebook storage and runtime limitations.

The workshop is a teaching prototype, not a deployed production service. It tests selected provenance and input/output contracts; human review still establishes whether claims follow from the cited evidence.
