# Result: cross-family-tail-consensus

**Planned comparison.** Target `ap`, reference `mc`, development bank `65e006afd99c` (50 tasks, 20 sources as the grader groups them). Prediction: mean AP at least 0.02 above `mc` and source wins on more than half the sources; wrong if it does not beat `mc`.

**Selector check.** `selector.py` (hash `cae3b1ce34ac`) applies one rule to every batch: rank-average each family, average the other families into a consensus, score each candidate by fractional top-5% overlap with that consensus. It reads only `batch.scores` and `batch.families`. It has no dataset names, row counts or per-dataset branches; the only constant is the 5% tail fixed in the prediction before grading. It abstains only on fewer than two families or non-finite scores, which never happened (50/50 ok). Accepted.

**Evidence.** Runner's grade: ledger run `cab003c885da`. Independent regrade from a rebuilt bank in this session: ledger run `5173a54a6f66`, which reproduces the same selector AP, regret and delta to every digit and links `prediction.md`.

**Outcome against the plan.** Both criteria are met on development. In `cab003c885da` the selector's mean AP is 0.443 against 0.413 for `mc`, delta_ap +0.030, AP regret 0.126 against 0.156. The regrade `5173a54a6f66` report gives 13 source wins out of 20. The prediction holds.

**Caveats, from the same report.** The 95% bootstrap interval for delta_ap over sources is [-0.001, +0.066], so the gain is not clearly separated from zero at 20 sources. The gain does not carry to ROC AUC: delta_auroc is +0.0004, interval [-0.017, +0.019]. A fixed choice beats it: `fixed/KNN/5` has mean AP 0.479 in `cab003c885da`, and the existing baselines `silhouette` (0.450) and `eag` (0.448) are level with it. So tail consensus improves on plain centrality but has not yet shown it chooses better than always picking KNN/5.

**Not too good.** Regret remains 0.126; no flag.

**Next.** A claim against `fixed/KNN/5` as reference would be the stronger test, and the AP-only gain suggests the tail focus matters for precision-type targets, which a later experiment could test directly.
