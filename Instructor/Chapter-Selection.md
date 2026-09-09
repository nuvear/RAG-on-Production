# Review of the nine chapters and two-hour lab selection

The supplied root contains nine corrected chapter PDFs, original chapter DOCX files, a validated Obsidian vault, nine existing Colab notebooks, code and validation reports, a student ZIP, a build pipeline and scratch dependencies. An additional `Chapter 10.docx` is present, but it has no corresponding corrected chapter or lab in the nine-chapter vault. It is outside this requested review. The existing student-lab guide estimates **420 minutes (seven hours)** for the nine complete labs, excluding teaching and breaks.

The short workshop is a new teaching package. Original PDFs, DOCX files, vault notes, chapter notebooks, ZIP and source code remain unchanged. The selection was based on the chapter topics, the existing notebook sections, recorded prerequisites and validation reports. Historical validation claims were treated as source evidence, not as proof that this new notebook runs.

## All nine chapters

| Chapter | Existing full lab estimate | Treatment in two-hour class | Reason |
|---|---:|---|---|
| 1. Introduction to RAG | 25 min | Orientation and Lab 3 | Establish retrieval plus generation and the need for evidence. Replace the full LangChain/LanceDB chain with inspectable Python. |
| 2. The Base RAG Stack | 60 min | Labs 1–3 | Keep chunking, embeddings, vector ranking and generation. Omit multiple parsers, PostgreSQL installation and alternative model clients. |
| 3. Scaling Your RAG Stack | 45 min | Labs 2–4 | Keep metadata eligibility, grounding and guardrail testing. Defer large rerankers, HHEM and framework-specific guardrail stacks. |
| 4. Deploying RAG to Production | 40 min | Debrief and code commentary | Discuss freshness, privacy, access control and latency. Embedding caching reduces repeated requests but is not the chapter's Redis/semantic answer cache. |
| 5. The RAG Platform | 35 min | Follow-on assignment | Vectara introduces a second account and a different platform workflow without advancing the core four experiments. |
| 6. Evaluating Your RAG Application | 30 min | Lab 4 | Keep labelled retrieval metrics and human generation review. Defer UMBRELA, paid LLM judges and wider metric suites. |
| 7. From RAG to AI Agents | 50 min | Follow-on assignment | Tool execution and multiple agent frameworks require separate safety and orchestration learning objectives. |
| 8. Multimodal RAG | 75 min | Follow-on assignment | Table, image and audio ingestion would consume much of the class in additional models and data setup. |
| 9. Knowledge-Enhanced RAG | 60 min | Follow-on assignment | Neo4j, large datasets and optional GraphRAG indexing require a separate session. The source lab itself treats paid indexing as optional. |

Times are the existing `RAG Vault/Notebooks/Student Labs.md` estimates, not new measured timings or guarantees. Full-lab estimates sum to 420 minutes. The two-hour schedule allocates 95 minutes to four labs, 10 to setup/orientation, 5 to break/catch-up and 10 to reporting/debrief.

## Four labs and acceptance evidence

| Lab | Concepts and implementation | Student acceptance evidence |
|---|---|---|
| 1, 20 minutes | Embedded text ingestion, stable source IDs, word-window chunking, overlap | Two parameter runs with chunk counts and an actual boundary observation. Metadata retained. |
| 2, 25 minutes | TF-IDF baseline, OpenAI embeddings, cosine ranking, unique-document results, current-policy eligibility | Rank comparison on a paraphrase and archived-policy result with filtering on/off. No promise that dense always wins. |
| 3, 25 minutes | OpenAI Responses, evidence-only instructions, JSON schema, source/quote validation, input limits and abstention | Supported renewal answer, oversized-input rejection, fabricated-ID/quote rejection and unsupported-question observation. |
| 4, 25 minutes | Precision, recall, MRR, usefulness vs abstention, adversarial evidence and human review | k=1/k=2 results, attack output checked for meaning, diagnosis and a proposed held-out test. |

## Deliberate adaptations

- Original synthetic library policies replace long book datasets so students can inspect every gold source. D03 is a controlled stale-policy counterexample.
- OpenAI `text-embedding-3-small` replaces the local MiniLM example for consistent key-based setup. `gpt-4.1-mini` supplies structured generation. No local model downloads are necessary.
- In-memory exhaustive ranking replaces a persistent vector database. The class demonstrates retrieval logic, not database deployment or approximate-index performance.
- Detailed functions are prewritten. Intermediate/expert Python students spend time modifying inputs, predicting outputs and testing invariants, rather than typing boilerplate. The code notes explain implementation details for pre-reading and later reuse.
- Guardrails are distributed through retrieval, generation and evaluation. They are not a fifth lab. Exact-quote validation verifies provenance syntax, not claim entailment. Prompt-injection resistance remains an observed behavior, not a guarantee.
- Reading every explanatory paragraph aloud does not fit the timetable. Assign the detailed notes as pre-reading/reference and focus live walkthroughs on the key functions and experiment variables.

## Traceability to local selection sources

Reviewed source locations in the supplied folder: `RAG Vault/Chapters/Chapter 01` through `Chapter 09` topic notes, the corresponding `RAG Vault/Notebooks/Chapter NN - Lab.ipynb` sections, `RAG Vault/Notebooks/Student Labs.md`, `RAG Vault/README.md`, and chapter validation report summaries. These paths identify the private source corpus and are not repository dependencies.
