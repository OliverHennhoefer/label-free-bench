# Handoff

Overwrite this file at the end of every session. Keep it under 40 lines.

**State.** Iteration 1 (runner) done on 2026-11-23.

- Development bank `65e006afd99c`: all 25 dev datasets, seeds 0 and 1, 50 tasks, basic bank. `.bench-runs` does not survive between cloud sessions; rebuild it with the `prepare` command in AGENTS.md (deterministic, same bank ID, a few minutes), then `run` the baselines before grading.
- Baseline ledger runs: `a1330c55e644` (ap) and `385fca4e060d` (auroc), reference iforest_default.
- `cross-family-tail-consensus`: `selector.py` implemented exactly as `prediction.md` describes and graded on development, target ap, reference mc: ledger run `cab003c885da`. Needs a verifier. The per-source wins and bootstrap interval are in the grade report (`.bench-runs/cross-family-tail-consensus-report/report.json`), which is not committed, so the verifier must regrade to see them.
- Still open, predictions committed, no selectors yet:
  - `synthetic-mix-ap`: refit pool, rank by AP against synthetic anomalies. Reference `fixed/KNN/5`.
  - `refit-overfit-gap`: refit pool, smallest train-vs-selection AUC gap. Reference `fixed/KNN/5`.

**Notes for runners.** `batch` gives families but not candidate names. Pool order is frozen in `bank.json` (`IForest/default, IForest/64, KNN/5, KNN/25, LOF/5, LOF/25, HBOS/5, HBOS/10`); refit experiments must verify each refit against `batch.scores` and abstain on mismatch. The cloud proxy blocks arxiv.org.

**Next action.** `python -m benchmarks next` should name the verifier for `cross-family-tail-consensus`: read ledger row `cab003c885da` and the selector code, regrade to read source wins and the interval, and write `result.md` plus a claim or negative row.
