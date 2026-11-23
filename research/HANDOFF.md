# Handoff

Overwrite this file at the end of every session. Keep it under 40 lines.

**State.** Setup (iteration 0) done on 2026-11-23.

- Development bank `65e006afd99c`: all 25 dev datasets, seeds 0 and 1, 50 tasks, basic bank (no extra views, so NPD, EM/MV and stability abstain). Rebuild with the `prepare` command in AGENTS.md; it is deterministic.
- Baseline ledger runs: `a1330c55e644` (target ap, reference iforest_default) and `385fca4e060d` (auroc). On ap the strongest fixed control is `fixed/KNN/5`; the best label-free baselines are `silhouette` and `eag`. Read the numbers from the ledger rows, not from here.
- Literature: 13 notes in `literature/`; `QUEUE.md` lists partial verifications and four new leads.
- Three open experiments, predictions committed, no selectors yet:
  - `synthetic-mix-ap`: refit pool, rank by AP against marginal-extreme and dependence-break synthetic anomalies. Reference `fixed/KNN/5`.
  - `refit-overfit-gap`: refit pool, prefer the smallest train-vs-selection score AUC gap. Reference `fixed/KNN/5`.
  - `cross-family-tail-consensus`: top-5% overlap with the other families' consensus, no refits. Reference `mc`.

**Notes for runners.** `batch` gives families but not candidate names. The pool order is frozen in `bank.json` (`IForest/default, IForest/64, KNN/5, KNN/25, LOF/5, LOF/25, HBOS/5, HBOS/10`); the two refit experiments must verify each refit against `batch.scores` and abstain on mismatch, as their predictions say. The cloud proxy blocks arxiv.org.

**Next action.** `python -m benchmarks next` should name the runner for one of the three experiments: implement `selector.py` exactly as its prediction describes and grade it on bank 65e006afd99c with the named target and reference.
