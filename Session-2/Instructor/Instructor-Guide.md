# Instructor guide · Session 2

**Two hours, four labs, intermediate/expert Python.** Use the PowerPoint to introduce each experiment, then leave the relevant task slide visible while students work in Colab. The guided notebooks include all required code and detailed explanations. This is an instructional lab, not a closed-book assessment; instructor answers are in the public teaching repository.

## Preparation before class

1. Complete Session 1 or review retrieval, evidence, source IDs, abstention and validation. Session 2 starts with a fresh runtime.
2. Send the English, Japanese or Simplified Chinese Colab link from the Session 2 README. Students keep English code and synthetic inputs; they may write reflections in their language.
3. Check the project can use `gpt-4.1-mini`, `text-embedding-3-small`, Files and Vector Stores. Only an OpenAI key is needed. No GPU, Vectara account, Neo4j server or framework install is required.
4. Rehearse setup, ingestion and cleanup with your teaching project shortly before class. Rates and permissions can change; check the official pricing link. The request cap is not a billing limit. Vector storage and original uploaded Files need deliberate lifecycle handling.
5. Keep [Recorded-Example-Run.md](Recorded-Example-Run.md) available when connectivity fails. Label recorded and live results separately. Do not automatically substitute a mock for the hosted platform.
6. Use only the supplied synthetic records. Each participant creates a named vector store and four small text files. Students must save their resource ledger and run cleanup before leaving.

## Timing and facilitation

| Clock | Activity | Instructor intervention |
|---|---|---|
| 00–10 | Setup | Get keys into Secrets and check both models. Pair after two failures. |
| 10–35 | Lab 5 | 5-minute walkthrough; 15-minute experiment; 5-minute review. Check readiness before searching. |
| 35–60 | Lab 6 | 5-minute loop walkthrough; 15-minute trace/control tests; 5-minute claim review. |
| 60–65 | Break | Save notebook and resource ledger. Do not use the break for another lecture. |
| 65–85 | Lab 7 | 4-minute concept; 12-minute comparison; 4-minute visual review. |
| 85–110 | Lab 8 | 5-minute path walkthrough; 15-minute hybrid/revocation experiments; 5-minute review. |
| 110–120 | Submission | Reflections, cleanup confirmation, export and one-minute pair reports. |

The 22 numbered steps are procedural checkpoints, not 22 separate coding assignments. Reference functions are supplied; students spend time predicting, changing conditions, inspecting evidence and explaining results. Save each output before rerunning a cell. The required change-and-rerun tasks are the equipment-policy question in Lab 5, LabB in Lab 6, the printing chart in Lab 7, and current-but-unverified E01 in Lab 8. Preserve both conditions in text worksheets and reflections because the report retains the last observation for each lab. All four reflection strings and four human-review strings are completed in the final review cell.

## Lab 5: expected reasoning

Four files must be indexed, with no failures. P01 allows at most **two hours per day**; P02's **four-hour** rule is archived. Unfiltered search may return P02 prominently. Its position is an observation, not an invariant. Current-only search must exclude it. The generated answer should reject a four-hour booking and cite current P01.

Ask: which operations moved to the provider? Expected examples: chunking, embedding, indexing, storage search. Which remain ours? Source accuracy, trusted eligibility attributes, identity/access rules, evidence instructions, evaluation, deletion. Do not claim this adapter reproduces Vectara's end-to-end feature set or hallucination correction. This lab explicitly calls Vector Stores search and then Responses generation; it does not invoke the hosted `file_search` tool.

`max_num_results=4` returns provider chunks and is not a unique-document guarantee. The source ID and status are attached attributes. A filter does not authenticate a tenant, and a citation does not verify semantic support.

## Lab 6: expected reasoning and a real failure to discuss

The useful live trace retrieves policy and Tuesday slots, in either order. P01 allows **at most** two hours. LabA's 10:00–11:00 slot is one hour and 14:00–16:00 is two hours: **either slot individually satisfies the duration limit**. Booking both would total three hours and violate the daily limit. For the changed LabB question, Tuesday has a single 09:00–10:00 one-hour slot that also satisfies the upper bound. No tool makes a booking.

The preparation run correctly retrieved P01 and `SLOTS-LabA` but said only the two-hour slot fits. This is a generation/interpretation failure: it confused an upper bound with an exact duration. It passed the source-ID contract. Use the unedited response in the recorded run to ask students to identify the unsupported inference. Do not fix the observed answer silently or award full semantic credit merely because `status='answered'`.

Deterministic controls: unknown `delete_booking` rejected, Friday rejected, zero-tool-budget test has no dispatch and no API calls. The zero-budget case uses a scripted test double and must be labelled as such. A live model can still choose poorly, repeat tools, omit useful evidence or produce an unsupported claim. The loop limits work; it does not prove security or task success.

Optional follow-on: add a deterministic duration predicate and tests for 1 hour, 2 hours and 3 hours. Preserve the original failure, use new questions for evaluation, and distinguish a verified scheduling rule from trusting the LLM to do all constraint reasoning.

## Lab 7: expected reasoning

The correct retrieved image for occupied seats is `IMG-SEATS`. The caption has no numeric count. Its PNG chart shows Monday 18, Tuesday **42**, Wednesday 27. `IMG-PRINTS` is a different subject and has Tuesday 16 print jobs. Students should check source, day, unit and value independently.

The caption-only condition should withhold the unsupported number. In the recorded run, the model abstained but incorrectly retained a citation; the application reported `output_rejected` and retained `raw_output`. This is a safe contract block, not an accepted abstention. The image-assisted answer returned 42 and cited `IMG-SEATS`. Neither a source-ID check nor a hash verifies numerical reading; students must look at the chart.

This is caption retrieval followed by vision. The changed printing question should select IMG-PRINTS and read Tuesday as 16 completed print jobs. Do not describe it as learned image-vector search, image-text joint embeddings, audio processing or PDF layout parsing. Those are follow-on topics in Chapter 8.

## Lab 8: expected reasoning

Valid path: **E01** `Team Orion —HAS_CERT→ Safety101`, then **E02** `Safety101 —QUALIFIES_FOR→ Projector X`. P03 supplies the access rule. E03 is revoked; E05 is current but unverified. Both must be excluded even though E04 could lead to Projector Y.

One hop yields no eligible device. Two hops yield Projector X. Both device descriptions may be semantically relevant; the graph constraint retains D-X only. The model should explain the team/certificate/device relationship with E01, E02 and P03, rather than just guessing the correct device name.

Revoking E01 in a deep copy leaves no valid path. An unknown team also has no path. In the changed condition, make E01 current but unverified instead of revoked; the path must still be excluded. A separate empty-evidence generation call should abstain, but record its actual status. Graph absence means the supplied graph does not establish access; it is not a universal proof that no real-world qualification exists.

The typed, bounded traversal is a curated in-memory exercise. Neo4j operations, schema evolution, entity linking, natural-language-to-Cypher, authenticating evidence, and Microsoft GraphRAG community indexing are outside this session. A trusted `verified` flag must come from governed ingestion in production, not a document's self-assertion.

## Recovery and cleanup

The resource ledger is updated after every successful create/delete operation. It is not an exactly-once transaction system. If creation times out after the server has accepted it but before the client records the ID, a retry can create another object. Use the unique `rag2-classroom-<run_id>` store name or `rag2_<run_id>_` file prefix in the same project's dashboard to reconcile only this run's objects. Never delete unrelated project resources.

If the VM is lost, restore the downloaded `session2_resources.json` in Colab Files **before** running setup. If no ledger survives, use the known run prefix in the project dashboard. The store expires after one day of inactivity, but original Files are separate. Verify their deletion explicitly.

If managed retrieval remains unavailable after two attempts, use paired/recorded Lab 5 evidence. In a fresh runtime, run setup and only the literal `POLICIES` assignment from the platform cell, then continue Labs 6–8. Mark Lab 5's provenance or incompletion clearly. Do not run the upload section repeatedly to recover unrelated variables.

Cleanup deletes only the ledger's store and files, and remains callable after the normal API budget cap. It is retryable and treats 404 as already absent. Students must confirm cleanup before leaving. Failed cleanup requires a preserved ledger and follow-up, not an unsupported claim of successful deletion.

## Assessment and handoff

Each lab: 1 point for concrete experiment evidence and 1 point for interpretation linked to that evidence, total 8. Strong interpretations name the failure layer and propose a held-out test. A well-diagnosed model error can earn both points.

Collect `rag_session2_report.json` plus the student's `.ipynb`. Check reflection and human-review fields; `complete=True` only tests nonempty strings. Check cleanup separately. Do not collect API keys. The student workbooks, technical notes and guided notebooks are available in English, Japanese and Simplified Chinese; the instructor deck is in English, as in Session 1.
