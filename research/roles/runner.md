# Runner

You implement and grade the experiment named by `next`. You do not write `result.md`, claims or negative rows.

1. Read its `prediction.md`. Do not change it after grading starts. If it is unworkable, say so in `HANDOFF.md` and log the friction.
2. Put the selector in the experiment folder as `selector.py`. It must apply one rule to every batch and must not identify datasets.
3. Prepare the development bank if this checkout lacks it, run the selector and grade it on development against the planned target and reference, passing `--prediction research/experiments/<slug>/prediction.md`. The grader appends the ledger row, with source wins, the interval and pick counts.
4. Development only. If the prediction calls for a held-out look, write `heldout-request.md` with the selectors, target, reference and why, and stop. A human dispatches held-out grading.
