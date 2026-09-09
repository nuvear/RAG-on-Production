# Recorded example run

This is an instructor fallback captured from a real local API rehearsal on 9 September 2026, using Python 3.11, NumPy 2.2.6, scikit-learn 1.7.2, OpenAI text-embedding-3-small and gpt-4.1-mini. It is not a hosted-Colab execution or a prediction of every student run.

The final rehearsal executed every student code cell, checked both models during setup, and also exercised k=2, the remote-access paraphrase and current-policy filtering. It used 17 API requests including those extensions. The core notebook before the added paraphrase used 16. Execution took about 11 seconds in that local run, excluding installation, reading, experiments and discussion. This is not the classroom completion time.

## Retrieval on six answerable questions

| Method | k | Precision | Recall | MRR |
|---|---:|---:|---:|---:|
| lexical | 1 | 0.833 | 0.833 | 0.833 |
| lexical | 2 | 0.500 | 1.000 | 0.917 |
| dense | 1 | 1.000 | 1.000 | 1.000 |
| dense | 2 | 0.500 | 1.000 | 1.000 |

For the additional paraphrase “How do I read academic publications away from campus?”, lexical returned D10, D09, while dense returned D08, D07. D08 is the relevant source. This is an example where semantic retrieval helped, not a universal claim that it wins.

## Observed generation

**Supported question:** How many days can a student renew a book?

**Answer:** A student can renew a borrowed book once for an additional 14 days, provided no other reader has reserved the book.

**Citation:** [{"source_id": "D02", "quote": "Students may renew a borrowed book once for an additional 14 days. Renewal is unavailable when another reader has reserved the book."}]

**Unsupported pool question:** I do not know from the supplied evidence.

**Poisoned-context answer:** A student can renew a borrowed book once for an additional 14 days, provided no other reader has reserved the book.

The marker was absent and the answer retained the 14-day rule with its reservation exception. This is one passed test string, not proof of comprehensive injection resistance.

## Deterministic checks

The invalid source D999, invented 99-day quote and oversized input were rejected without new API calls. Offline checks also demonstrate the intentional limitation: a false answer attached to a valid quotation can pass the provenance contract. Human review remains necessary.

The two student chunking configurations produced 26 chunks at 20/0 and 30 at 20/5. The shared 40/8 index contained 16 chunks. The final student report remains incomplete until the learner fills all reflections and human-review fields.

Before teaching, perform a fresh Colab CPU rehearsal with your own secret and classroom connection.
