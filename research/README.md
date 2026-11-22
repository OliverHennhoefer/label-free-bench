# Research memory

**Question.** With only known-normal data, how close can a selector come to the detector a labeled evaluation would choose for a named target (ROC AUC, AP, PR AUC or F1)? [Protocol](../benchmarks/README.md#normal-only-protocol)

This file is the index. Read it first and load entries only when you need them. Keep it under 150 lines: one line per entry, details in the entry.

## Layout

| Path | Holds | Written by | Limit |
| --- | --- | --- | --- |
| `ledger.jsonl` | One row per grade: run ID, split, bank, commits, selector hashes, results | The grader only | Append-only |
| `claims/C-NNN.md` | One atomic claim, its scope, status and ledger evidence | Verifier, gardener | 150 words |
| `negative.md` | One row per dead end: mechanism tested, scope, why it failed, when to reopen | Verifier, gardener | 60 words a row |
| `experiments/<slug>/` | `prediction.md` (committed before grading), `result.md`, selector code | Researcher | 250 and 350 words |
| `literature/<yyyy-mm-dd>-<slug>.md` | One source: what it establishes, setting, relevance, link, date checked | Researcher | 200 words |
| `literature/QUEUE.md` | Sources to read and verify, not yet notes | Anyone | 400 words |
| `loop.jsonl` | One row per role session: role, experiment, session, summary, friction | `benchmarks log` | Append-only |
| `loop.toml`, `LOOP.md`, `roles/` | Loop cadence, session steps and role contracts | A human | |
| `HANDOFF.md` | Current state and the next action | Last session | 40 lines, overwritten |
| `archive/` | Retired or merged entries | Gardener | Not read by default |
| `../observer/` | Process reports for a human | Observer | 300 words, not read by research roles |

`python -m benchmarks check-memory` enforces the limits and checks that every claim's evidence exists in the ledger. CI runs it.

## Rules for entries

- **Numbers come from the ledger.** Cite run IDs. Never copy a number from a summary or from memory.
- **Claims are atomic and scoped.** One statement, the conditions it covers, status `open`, `supported`, `refuted` or `superseded by C-NNN`.
- **Negative results stay short and specific.** Record the mechanism and scope that failed, not "X does not work", plus what would justify reopening it.
- **IDs are permanent.** Merge or retire by marking `superseded by` and moving the old file to `archive/` with `git mv`. Never delete evidence.
- **Literature on demand.** Search when a hypothesis is proposed and again before any novelty claim. Record the query and date in the note.

## Open questions

Protocol questions, not hypotheses. Each changes the frozen protocol, so adopting one needs a new bank and a recorded decision.

- Q1. Widen the pool beyond 8 candidates in 4 families, for example adding PCA and One-Class SVM, and test sensitivity to removing or duplicating a family.
- Q2. Report normalized regret, (oracle − chosen) / (oracle − worst), next to raw regret.
- Q3. Count a failed or abstaining selector as the worst choice, next to the default-Isolation-Forest fallback.
- Q4. Test robustness to anomaly prevalence by reweighting classes to 1%, 5% and 10%.
- Q5. Add a canary: a renamed, row- and column-permuted copy of a development dataset must receive the same ranking.

## Claims

None yet.

## Data exposure

Record every source that has produced graded feedback outside development.

- **label-free-bench v1 (July 2026, commit `0841a31` in this repository's history)** published aggregate graded results that included validation sources vowels and musk, and test sources landsat, satellite, satimage-2, shuttle, fault and PageBlocks. No per-dataset outcomes were published. Disclose this in any paper.
- **Validation and test looks under this protocol:** none.
