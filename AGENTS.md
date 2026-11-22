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
python -m benchmarks next     # which role this session plays
python -m benchmarks log --role <role> --slug <slug> --summary "..."
python -m benchmarks check-memory
python -m pytest
```

## Hard rules

1. **Never read grading data.** Do not open a `private/` directory, a key directory, `truth.json` or raw ADBench files. Selectors see only the batch they are given. The runner's static screen rejects file, network and process access, but it is a tripwire, not a sandbox.
2. **Numbers come only from `research/ledger.jsonl`.** The grader writes it. Never edit it, and cite run IDs for every number you report.
3. **Development is free. Validation and test are not.** Commit `prediction.md` before any validation or test look and pass it with `--prediction`. Validation returns rounded aggregates. Test returns only promote or hold, and is for one or two finalists chosen in advance.
4. **Name the target and reference before grading.** Report the comparison you planned, including when it fails.
5. **Do not patch `labelfree` here.** Record a suspected metric bug as a claim, fix it in [label-free](https://github.com/OliverHennhoefer/label-free), then move the commit pin in `requirements.txt`.
6. **Stop and flag a human** when a result looks too good, a rule blocks you, or the protocol itself seems wrong. Start `HANDOFF.md` with a line beginning `FLAG`, which halts the loop. Never work around a rule.
7. **Aim for one rule that generalizes** across numeric-only tabular data. A selector that branches on dataset identity or is tuned to particular datasets does not count. Held-out sources stay unseen until a finalist's look.
8. **The loop never dispatches held-out grading.** Write `heldout-request.md` in the experiment folder; a human dispatches it.

## Research loop

Each session plays one role, chosen by `python -m benchmarks next`. See `research/LOOP.md` for the session steps and `research/roles/` for each role's contract. Do not read `observer/`.

1. **Propose.** Write the mechanism and a discriminating prediction from first principles before reading `claims/` or `negative.md`. Then read them and state for each relevant entry whether you adopt or reject it, and why. Search the literature for the mechanism. Commit `prediction.md`.
2. **Implement and run.** Put the selector in the experiment folder. Grade it against the planned reference on development.
3. **Verify.** A fresh session reads the ledger rows and the selector code, not the proposer's summary, and writes `result.md`, then a claim or a negative row.
4. **Garden.** Follow `research/GARDENING.md` on the cadence in `research/loop.toml` or when `check-memory` fails.
5. **Observe.** An outside observer reports on the process to a human. It never takes part in the research.

Keep several mechanistically different ideas alive at once rather than tuning one. Before a held-out look, check that a fresh session can reimplement the method from a description of at most 150 words and reproduce its development result.

Entry formats and size limits are in `research/README.md`. End every session with `python -m benchmarks log`, then overwrite `research/HANDOFF.md` unless you are the verifier or observer.
