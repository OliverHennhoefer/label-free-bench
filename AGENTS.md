# Lab rules

The question: with only known-normal data, how close can a selector come to the detector a labeled evaluation would choose for a named target? Start every session with `research/README.md`, then `research/HANDOFF.md`.

## Commands

Run from the repository root after `python -m pip install -r requirements.txt` (Python 3.12).

```sh
python -m benchmarks prepare --download --raw-dir .bench-runs/raw --split dev \
  --seeds 0 1 --out .bench-runs/dev --key-dir ../lab-keys/dev
python -m benchmarks run --bank .bench-runs/dev/public --out .bench-runs/baselines
python -m benchmarks run --bank .bench-runs/dev/public --selector research/experiments/<slug>/selector.py --out .bench-runs/<slug>
python -m benchmarks grade --truth ../lab-keys/dev --submissions .bench-runs/baselines .bench-runs/<slug> \
  --target ap --reference iforest_default --out .bench-runs/<slug>-report
python -m benchmarks check-memory
python -m pytest
```

Validation and test looks run only in CI, through the [held-out workflow](.github/workflows/heldout.yml). It builds the held-out bank from a secret-keyed row sample, runs your committed selectors there as an offline, unprivileged user, grades them and pushes a signed ledger row to `main`. Locally, `prepare` refuses those splits. See [Held-out looks](benchmarks/README.md#held-out-looks-in-ci).

## Hard rules

1. **Never read grading data.** Do not open a `private/` directory, a key directory, `truth.json` or raw ADBench files. Selectors see only the batch they are given. The runner's static screen rejects file, network and process access, but it is a tripwire, not a sandbox.
2. **Numbers come only from `research/ledger.jsonl`.** The grader writes it. Never edit it, and cite run IDs for every number you report.
3. **Development is free. Validation and test are not.** Commit the selector and a `prediction.md` that names its path before any validation or test look, then request the look through the held-out workflow. Validation returns rounded aggregates. Test returns only promote or hold, takes at most two finalists, and each selector version gets one test look. Never fetch validation or test ADBench files, and never write a selector that recognizes a dataset.
4. **Name the target and reference before grading.** Report the comparison you planned, including when it fails.
5. **Do not patch `labelfree` here.** Record a suspected metric bug as a claim, fix it in [label-free](https://github.com/OliverHennhoefer/label-free), then move the commit pin in `requirements.txt`.
6. **Stop and flag a human** when a result looks too good, a rule blocks you, or the protocol itself seems wrong. Write it at the top of `HANDOFF.md`. Never work around a rule.

## Research loop

Use separate sessions, or separate subagents with fresh context, for these roles.

1. **Propose.** Write the mechanism and a discriminating prediction from first principles before reading `claims/` or `negative.md`. Then read them and state for each relevant entry whether you adopt or reject it, and why. Search the literature for the mechanism. Commit `prediction.md`.
2. **Implement and run.** Put the selector in the experiment folder. Grade it against the planned reference on development.
3. **Verify.** A fresh session reads the ledger rows and the selector code, not the proposer's summary, and writes `result.md`, then a claim or a negative row.
4. **Garden.** Follow `research/GARDENING.md` after every 10 graded experiments or when `check-memory` fails.

Keep several mechanistically different ideas alive at once rather than tuning one. Before a held-out look, check that a fresh session can reimplement the method from a description of at most 150 words and reproduce its development result.

Entry formats and size limits are in `research/README.md`. End every session by overwriting `research/HANDOFF.md`.
