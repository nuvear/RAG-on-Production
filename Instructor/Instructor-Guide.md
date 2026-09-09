# Instructor guide: two hours, four labs

Teach with `RAG-2h-Labs.pptx` and the student notebook side by side. The audience is intermediate-to-expert in Python. The learning challenge is interpreting RAG behavior and designing tests. Detailed explanations are already present beside the code and in `Student/Python-Code-Notes.md`.

## Before class

**One day before:** send the repository link, ask students to check their Google/Colab access and funded OpenAI project, and assign the setup and Python notes as optional pre-reading. Each student should create a Colab Secret called `OPENAI_API_KEY` and enable notebook access. Confirm access to both configured model IDs. Do not distribute a shared key in the repository or slides.

**Instructor rehearsal:** open the GitHub Colab link on a fresh free CPU runtime, save a copy, enable the secret and run through all cells. Test both k values, the filter on/off, unsupported questions and the poisoned-context test. Save a completed reference notebook for a network fallback. Rehearse from the actual classroom network when possible. A successful local API run does not verify a hosted Colab runtime, account permissions or classroom timing.

**Ten minutes before:** open the PowerPoint, your student notebook and this guide. Start with a fresh runtime so cache state is clear. Keep a working executed reference copy available privately. Pair students if account setup fails twice. Do not spend the whole class debugging one account.

## Minute-by-minute facilitation

| Clock | Slides | Instructor action | Student activity / checkpoint |
|---|---|---|---|
| 00–03 | 1–2 | State the outcome and schedule. Ask learners to start setup. | Open notebook and configure secret. |
| 03–07 | 3–4 | Explain why the nine chapters become four focused labs. | Identify deferred topics. |
| 07–10 | 5–6 | Confirm API setup and show the fictional corpus. | READY status and D02/D03 conflict. |
| 10–13 | 7–8 | Walk through stride, source ID and retained metadata. | Predict boundary behavior. |
| 13–25 | 7–8 | Circulate. Ask for actual repeated phrases. | Run 20/0 and 20/5, record counts and boundary observation. |
| 25–30 | 9 | Invite one trade-off explanation. | Save Lab 1 reflection and reset shared chunks. |
| 30–34 | 10–11 | Explain matrix shape, cosine and unique-document ranking. | Predict lexical vs dense results. |
| 34–50 | 10–11 | Prompt both paraphrase and stale-policy experiments. | Compare rankings, toggle eligibility and record IDs. |
| 50–55 | 12 | Check that everyone can distinguish filtering from relevance. | Lab 2 reflection. |
| 55–60 | 13 | Break/catch-up. Help a stalled pair rerun preceding cells. | Save notebook. |
| 60–65 | 14–15 | Walk through the input contract, instructions and output validator. | Trace where each guardrail executes. |
| 65–80 | 14–16 | Ask students to inspect exact quotes and omissions. | Generate renewal answer, unsupported question and deterministic rejection tests. |
| 80–85 | 17 | Ask what a valid quote cannot prove. | Lab 3 reflection and human evidence check. |
| 85–90 | 18–19 | Compute the paper metric example together. | Verify denominators and rank indexing. |
| 90–98 | 18–19 | Compare k=1 and k=2 without claiming dense always wins. | Run metric table and record both settings. |
| 98–105 | 20 | Explain the poisoned-context canary and its narrow scope. | Run the attack and inspect factual meaning. |
| 105–110 | 20 | Separate usefulness, abstention and injection observations. | Complete human review and Lab 4 reflection. |
| 110–116 | 21–22 | Debrief production gaps and collect evidence. | Export JSON and save/download notebook. |
| 116–120 | 22 | Invite one proposed improvement and held-out test per pair. | Final discussion. |

Slides 23–24 are reference/troubleshooting appendices. Do not present them as additional timed lessons.

A [recorded live example](Recorded-Example-Run.md) is included for instructor comparison and outage discussion. It is explicitly a local run.

## Expected findings and teaching responses

**Lab 1:** overlap trades duplicated text for boundary coverage. Read the actual D02 windows. The default 40/8 index is a common baseline, not a tuned production recommendation. All chunks must retain IDs and status.

**Lab 2:** D08 supports remote journals. D03 is archived and must be absent with the default filter. Dense may tie or lose to lexical on this tiny corpus. Avoid describing cosine as confidence or implying that current metadata proves truth. Filtering here is not server-side authorization.

**Lab 3:** D02 supports 14 additional days, once, unless reserved. The generated answer should cite exact supplied text; students must identify omissions. D999 and the fabricated 99-day quote must be rejected. Oversized questions must fail before a new API call. The pool question should abstain, but record the actual result. A fabricated answer attached to a real quote illustrates the gap between evidence-reference validation and entailment.

**Lab 4:** the paper example is precision .5, recall .5, RR 1. With one gold source, correct retrieval at k=2 has precision .5. A perfect score on six simple questions is a starting point. The attack test injects a string after retrieval, deliberately isolating generation behavior. Marker absence is not sufficient: inspect whether the answer still states the correct policy. Blocking or refusal is different from a useful grounded answer.

## Interventions and fallback

| Symptom | Action |
|---|---|
| Secret missing / 401 | Recheck secret name and notebook-access switch. Never paste a key into a shared cell. |
| 403 / 404 | Verify that the key's project can access the configured model. Instructor tests any replacement before changing classroom instructions. |
| 429 | Check quota/rate limits and pause. Retry manually once after resolving the cause. |
| Timeout | Check the connection. A timeout can be ambiguous about whether a request was processed. There is no automatic retry loop. |
| Import error after install | Restart the runtime once and rerun from setup. Do not mix arbitrary dependency upgrades mid-class. |
| NameError after reconnect | Rerun setup and Labs 1–2. The in-memory index/cache was lost. |
| `output_rejected` | Inspect supplied chunks. Exact-quote enforcement can reject paraphrased quotations. Treat this as a measurable false block, not a reason to remove the guardrail silently. |
| Running 5 minutes behind | Skip optional new-question extensions. Retain all four core labs, the guardrail rejection tests and one adversarial test. |
| Persistent API outage | Pair with a working account or use the instructor's saved run. Students can still execute chunking, lexical retrieval, metric arithmetic and deterministic validators. Label recorded API outputs as recorded, not live. |

## Assessment and handover

Collect the changed notebook and `rag_workshop_report.json` through your normal course channel. Award 2 points per lab: 1 for experiment evidence, 1 for an interpretation tied to evidence. Full score is 8. Correctly diagnosed failures receive credit. Empty reflection fields leave `complete=False`.

The instructor notebook includes reference interpretations in a public repository. This is a formative workshop, not an answer-key-protected exam. Ask each learner to add their own observations. Use Chapters 3, 4 and 6 for the next production session, then Chapters 5, 7, 8 and 9 for specialist tracks.
