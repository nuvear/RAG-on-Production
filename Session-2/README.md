# Session 2 · Four more RAG labs

**Two hours · Labs 5–8 · OpenAI API keys · CPU Colab · Intermediate/expert Python**

Continue with a managed RAG platform, an AI agent, multimodal evidence and a knowledge graph. Each workbook includes concepts, 22 numbered steps, detailed Python explanations, observation tables, recovery and submission instructions. One notebook preserves state across the four labs; Session 1 need not be running.

| Language | Hands-on workbook | Colab | Detailed Python notes |
|---|---|---|---|
| English | [Workbook](Student/en/Hands-On-Workbook.md) | [Open](https://colab.research.google.com/github/nuvear/RAG-on-Production/blob/main/Session-2/Student/en/RAG_Session2.ipynb) | [Code notes](Student/en/Python-Code-Notes.md) |
| 日本語 | [ワークブック](Student/ja/Hands-On-Workbook.md) | [開く](https://colab.research.google.com/github/nuvear/RAG-on-Production/blob/main/Session-2/Student/ja/RAG_Session2.ipynb) | [Python解説](Student/ja/Python-Code-Notes.md) |
| 简体中文 | [实操手册](Student/zh-CN/Hands-On-Workbook.md) | [打开](https://colab.research.google.com/github/nuvear/RAG-on-Production/blob/main/Session-2/Student/zh-CN/RAG_Session2.ipynb) | [Python详解](Student/zh-CN/Python-Code-Notes.md) |

Translated prose includes English technical terms in brackets. All 12 executable cells, comments, synthetic data and test inputs are identical across languages. Students may write reflections in their preferred language.

| Clock | Lab | Student result |
|---|---|---|
| 00–10 | Setup | Preflight and resource recovery plan |
| 10–35 | 5. RAG platform | Hosted ingestion, filtered retrieval and cited generation |
| 35–60 | 6. AI agent | Two read-only tools, live trace, bounded execution tests |
| 60–65 | Break | Saved work |
| 65–85 | 7. Multimodal RAG | Caption retrieval and actual image-input comparison |
| 85–110 | 8. Knowledge graph | Typed two-hop traversal, hybrid candidates and revocation |
| 110–120 | Review and cleanup | JSON, notebook and resource deletion confirmation |

## Instructor materials

- [PowerPoint presentation](Instructor/RAG-Session2-Labs.pptx)
- [Instructor guide and answer guidance](Instructor/Instructor-Guide.md)
- [Recorded live example, including model failures](Instructor/Recorded-Example-Run.md)
- [Source-chapter mapping and official references](REFERENCES.md)

Lab numbering continues Session 1. The source chapters are **5, 7, 8 and 9**, respectively. OpenAI managed retrieval adapts the platform chapter's concepts so no Vectara account is required. The graph is an in-memory implementation, not a Neo4j or Microsoft GraphRAG deployment.

## Run and clean up

Open a Colab edition, select Copy to Drive, use Python 3 CPU and configure `OPENAI_API_KEY` in Secrets. The notebook uses `gpt-4.1-mini`, `text-embedding-3-small` and standard-library HTTP; IPython supplies notebook image display. Local Jupyter users set the environment key and may set `SESSION2_DATA_DIR` to the local `data` folder.

API and vector storage usage may incur charges. The current-runtime cap of 100 ordinary requests and 600 output tokens per generation is not an account spending limit. Setup preserves the same-runtime counter. Download `session2_resources.json` after ingestion. **Run cleanup before leaving:** delete the workshop vector store and original uploaded Files; store expiry does not replace separate Files deletion. No classroom credentials are distributed.

## Rebuild and verify

From the repository root:

```bash
python Session-2/scripts/build_notebooks.py
python Session-2/scripts/check_session2.py
```

Edit `code/*.py` for the shared implementation and `Student/<language>/Hands-On-Workbook.md` for prose. The builder regenerates all notebooks and full code notes. Offline checks exercise dispatch, budgets, graph eligibility, output contracts, cleanup, asset hashes and code identity without an API key. PNG creation is an authoring step requiring matplotlib 3.10.8; students download the finished images and do not install matplotlib.

The notebooks ship without outputs. Live rehearsal observations are preserved separately and are not promises about future model behavior. Session 1 materials remain unchanged.
