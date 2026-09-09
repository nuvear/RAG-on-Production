# RAG on Production: a two-hour Colab workshop

Four hands-on labs for intermediate-to-expert Python programmers. Use OpenAI embeddings and generation to build a small evidence-grounded assistant, then test retrieval and guardrails.

[![Open student workbook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nuvear/RAG-on-Production/blob/main/Student/RAG_2h_Student.ipynb)

## Start here

1. **Instructor:** read [Instructor-Guide.md](Instructor/Instructor-Guide.md), then use [the PowerPoint](Instructor/RAG-2h-Labs.pptx).
2. **Students:** open the Colab workbook above, save your own copy and configure `OPENAI_API_KEY` in Colab Secrets. The code explains itself in text cells and in the [detailed Python notes](Student/Python-Code-Notes.md).
3. **Course designer:** read the [nine-chapter review and lab selection](Instructor/Chapter-Selection.md).

| Clock | Session | Student output |
|---|---|---|
| 00–10 | Setup and orientation | Working API connection |
| 10–30 | Lab 1: chunking and source metadata | Boundary experiment |
| 30–55 | Lab 2: retrieval and current-policy filtering | Ranked-source comparison |
| 55–60 | Break and catch-up | Saved notebook |
| 60–85 | Lab 3: grounded answers and guardrails | Cited answer and rejection tests |
| 85–110 | Lab 4: evaluation and adversarial testing | Metrics and failure diagnosis |
| 110–120 | Report and discussion | Saved notebook and JSON report |

One notebook contains all four labs. This preserves the corpus, embeddings, cache and experiment state in one runtime. Use the table of contents to navigate. The instructor workbook contains reference interpretations and is publicly accessible in this teaching repository; it is not intended as a secure exam answer key.

## Requirements

Use a Google account, free Colab Python 3 CPU runtime, internet access and a funded OpenAI API key. The models are `text-embedding-3-small` and `gpt-4.1-mini`. The project must allow both models. No GPU, database server, model download or Drive mount is required.

API usage is paid. The workbook limits attempted calls to 40 per kernel state and generation to 400 output tokens per response. It caches embeddings in memory. These are workshop controls, not an account spending cap. A USD 1 planning allowance per student is a reserve, not a forecast. Review [current model prices](https://developers.openai.com/api/docs/pricing) and account controls before class. Never add keys to code, notebook output, screenshots or Git.

## Materials

- [Student workbook](Student/RAG_2h_Student.ipynb) and [detailed Python notes](Student/Python-Code-Notes.md).
- [Instructor workbook](Instructor/RAG_2h_Instructor.ipynb), [facilitation guide](Instructor/Instructor-Guide.md) and [PowerPoint](Instructor/RAG-2h-Labs.pptx).
- [Original fictional library data](Student/library_corpus.json), also embedded in each notebook.
- [Recorded live example](Instructor/Recorded-Example-Run.md) for instructor comparison.
- [Chapter selection](Instructor/Chapter-Selection.md) and [source references](REFERENCES.md).

## Guardrails in scope

Current-policy filtering, input length checks, instructions separating evidence from commands, structured response fields, source-ID allowlisting, exact-quote validation, abstention handling, and a prompt-injection canary test. These controls do not prove factual entailment, implement tenant authorization, or provide a complete moderation/PII system. Students inspect those gaps explicitly.

## Local use and checks

Create a Python 3.11+ virtual environment and install `requirements.txt`. Set `OPENAI_API_KEY` in your shell, then open the notebook in Jupyter. The notebooks do not read a repository `.env` file. Run `python scripts/check_notebooks.py` for offline syntax, guardrail and metric checks. It makes no API calls. Local package installation and API execution are separate from the Colab UI rehearsal described in the instructor guide.

## Source scope

The workshop adapts concepts from Chapters 1–4 and 6 of Mendelevitch & Bao, *Hands-On RAG for Production*, using the user's corrected nine-chapter vault as the selection source. This repository contains newly written exercises, explanations and fictional data, rather than copies of the source book chapters or its full code packages. The full nine-chapter course remains the next step.
