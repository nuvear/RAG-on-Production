# Recorded preparation run · 9 September 2026

This is an actual local Python rehearsal of the shared code, using OpenAI APIs and the supplied synthetic assets. It is not an execution inside a signed-in Colab runtime, not a model-quality benchmark, and not a student submission. Future responses may differ. Use this record to discuss evidence, including failures; do not label it as your own live result.

The complete run took **35.71 seconds** of execution time, excluding classroom reading, prediction, discussion and worksheet completion. It made **31 HTTP requests**, including cleanup. All requests in the completed run returned successfully. This is an observed count, not a spending cap or predicted invoice. Endpoint counts: `{'embeddings': 3, 'responses': 9, 'vector_stores': 11, 'files': 8}`.

## Lab 5

All four files indexed. Unfiltered IDs: `P02, P01, P03, P04`. Filtered IDs: `P01, P03, P04`. The archived P02 ranked first without filtering and was absent with current-only filtering.

```json
{
  "answer": "No, a student may only book a makerspace room for up to two hours per day.",
  "abstain": false,
  "sources": [
    "P01"
  ],
  "guardrail_status": "contract_passed"
}
```

## Lab 6: valid citations, wrong constraint interpretation

The following is the actual response, including its error:

```json
{
  "status": "answered",
  "answer": {
    "answer": "The maximum room booking duration allowed per day is two hours. For LabA on Tuesday, the available slots are 10:00-11:00 (1 hour) and 14:00-16:00 (2 hours). Only the 14:00-16:00 slot fits the maximum booking duration of two hours.",
    "abstain": false,
    "sources": [
      "P01",
      "SLOTS-LabA"
    ]
  },
  "trace": [
    {
      "round": 1,
      "tool": "search_policy",
      "arguments": {
        "query": "maximum room booking duration"
      },
      "source_ids": [
        "P01",
        "P03"
      ],
      "seconds": 0.0003
    },
    {
      "round": 2,
      "tool": "get_slots",
      "arguments": {
        "room": "LabA",
        "day": "Tuesday"
      },
      "source_ids": [
        "SLOTS-LabA"
      ],
      "seconds": 0.0
    }
  ]
}
```

**Instructor diagnosis:** both one-hour and two-hour slots individually fit an upper bound of two hours. Only selecting the two-hour slot confuses “at most” with “exactly.” Both slots together total three hours and must not be booked on the same day under P01. The retrieved facts and source IDs were valid; the answer's interpretation was wrong. The function loop and source-ID check do not verify constraint reasoning.

Deterministic application-control tests (scripted, not model-generated):

```json
{
  "forbidden_tool": "rejected",
  "invalid_day": "rejected",
  "zero_tool_budget": "tool_budget_exhausted",
  "api_calls_added": 0
}
```

## Lab 7: blocked caption-only output, supported visual answer

```json
{
  "ranked_ids": [
    "IMG-SEATS",
    "IMG-PRINTS"
  ],
  "selected_id": "IMG-SEATS",
  "text_only": {
    "guardrail_status": "output_rejected",
    "reason": "Abstention must not cite evidence.",
    "raw_output": {
      "answer": "The exact number of occupied seats recorded on Tuesday is not specified in the provided evidence.",
      "abstain": true,
      "sources": [
        "IMG-SEATS"
      ]
    }
  },
  "with_image": {
    "answer": "42",
    "abstain": false,
    "sources": [
      "IMG-SEATS"
    ],
    "guardrail_status": "contract_passed"
  }
}
```

The caption-only model withheld the number but retained a citation while abstaining; the contract rejected that output. This is `output_rejected`, not an accepted `abstained` result. The image condition read 42 occupied seats on Tuesday from IMG-SEATS. Review the actual chart, not just the allowed source ID.

## Lab 8

Valid path: E01 → E02 → Projector X. One hop returned no device. Vector candidates were `D-Y, D-X`; graph constraints retained `D-X`.

```json
{
  "answer": "Team Orion may use Projector X because they have the Safety101 certificate, which qualifies them for Projector X under the equipment access policy.",
  "abstain": false,
  "sources": [
    "E01",
    "E02",
    "P03"
  ],
  "guardrail_status": "contract_passed"
}
```

Revocation and no-evidence result:

```json
{
  "after_revocation": [],
  "unknown_team": [],
  "no_evidence_answer": {
    "answer": "I do not know from the supplied evidence.",
    "abstain": true,
    "sources": [],
    "guardrail_status": "abstained"
  }
}
```

## Cleanup and student completion

```json
{
  "results": [
    {
      "kind": "vector_store",
      "deleted": true
    },
    {
      "kind": "file",
      "source_id": "P01",
      "deleted": true
    },
    {
      "kind": "file",
      "source_id": "P02",
      "deleted": true
    },
    {
      "kind": "file",
      "source_id": "P03",
      "deleted": true
    },
    {
      "kind": "file",
      "source_id": "P04",
      "deleted": true
    }
  ],
  "complete": true
}
```

All resources created by this completed run were deleted. The earlier interrupted development run was also cleaned up. The rehearsal left student reflection fields blank, so its submission flag was correctly `False`; generated outputs do not constitute student reflection or human review.

Offline tests additionally cover invalid dispatch arguments, unknown tools, tool and round budgets, revoked and unverified graph edges, source contracts, scoped cleanup, image hashes, and identical executable cells in all three language notebooks. These are application checks, not evidence of comprehensive agent security or production performance.

## Change-and-rerun verification

A separate local live run exercised all four changed conditions using the shared cells with only the workbook-prescribed edits. It took 36.87 seconds and 31 requests including cleanup. These are a separate run, not additional rows from the baseline above.

### Equipment qualification question

```json
{
  "answer": "A team needs a current, verified certificate that qualifies it for the specific device in order to use the equipment.",
  "abstain": false,
  "sources": [
    "P03"
  ],
  "guardrail_status": "contract_passed"
}
```

### LabB agent question

```json
{
  "status": "answered",
  "answer": {
    "answer": "The maximum room booking duration allowed per day is two hours according to the makerspace policy. On Tuesday, LabB has one available slot from 09:00 to 10:00, which fits within the maximum two-hour booking duration.",
    "abstain": false,
    "sources": [
      "P01",
      "SLOTS-LabB"
    ]
  },
  "trace": [
    {
      "round": 1,
      "tool": "search_policy",
      "arguments": {
        "query": "maximum room booking duration"
      },
      "source_ids": [
        "P01",
        "P03"
      ],
      "seconds": 0.0002
    },
    {
      "round": 2,
      "tool": "get_slots",
      "arguments": {
        "room": "LabB",
        "day": "Tuesday"
      },
      "source_ids": [
        "SLOTS-LabB"
      ],
      "seconds": 0.0
    }
  ]
}
```

### Printing chart question

```json
{
  "ranked_ids": [
    "IMG-PRINTS",
    "IMG-SEATS"
  ],
  "selected_id": "IMG-PRINTS",
  "text_only": {
    "guardrail_status": "output_rejected",
    "reason": "Abstention must not cite evidence.",
    "raw_output": {
      "answer": "The provided evidence describes a bar chart of completed print jobs on Monday, Tuesday, and Wednesday but does not provide any specific numerical values for the number of completed print jobs on Tuesday.",
      "abstain": true,
      "sources": [
        "IMG-PRINTS"
      ]
    }
  },
  "with_image": {
    "answer": "16 completed print jobs were recorded on Tuesday.",
    "abstain": false,
    "sources": [
      "IMG-PRINTS"
    ],
    "guardrail_status": "contract_passed"
  }
}
```

### E01 current but unverified

```json
{
  "after_revocation": [],
  "unknown_team": [],
  "no_evidence_answer": {
    "answer": "I do not know from the supplied evidence.",
    "abstain": true,
    "sources": [],
    "guardrail_status": "abstained"
  }
}
```

The graph output field keeps the reference code name `after_revocation`; in this changed run, the applied change was `verified=False`, not revocation. Both conditions returned no eligible path. The printing image answered 16. The LabB answer correctly accepted the one-hour slot. All resources created by this variation run were deleted.
