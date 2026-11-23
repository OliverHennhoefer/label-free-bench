# Prediction: synthetic-mix-ap

**Mechanism (written first,).** A detector's AP on real anomalies is approximated by its AP on synthetic anomalies built from normals, if the synthetic types span the ways real tabular anomalies differ. Use two equally weighted types: marginal extremes (a copy of a selection normal with a few features pushed just beyond the fitting 1st/99th percentiles) and dependence breaks (each feature permuted independently across selection rows). The selector refits the public pool on `views["train"]` by matrix position, confirms each refit reproduces `batch.scores` (Spearman ≥ 0.99, else abstain for that task), scores the synthetic rows and ranks candidates by mean AP of selection normals against synthetic rows at 10% prevalence.

**Discriminating prediction.** Target `ap`, reference `fixed/KNN/5`, development bank 65e006afd99c. The mechanism predicts a mean AP gap at most 0.090 (no worse than KNN/5 in run a1330c55e644) and a best-pick rate above KNN/5's 26.2%. It is wrong if the selector scores below `eag` (0.448, a1330c55e644), the best existing label-free baseline. Report the per-type pick split: if dependence breaks alone drive picks toward LOF/KNN and marginal extremes toward HBOS, say which one carried the result.

**Prior entries.** No claims or negative rows exist yet; nothing to adopt or reject.

**Literature (2026-11-23).** Queries: "arxiv 2310.10461 model selection anomaly detection synthetic anomalies"; "tabular anomaly detection model selection synthetic anomalies feature permutation dependency breaking validation". SWSA (Fung 2023) supports the mechanism for images and fails where synthetic anomalies mismatch real ones; Röchner 2025 motivates the marginal-extreme type.

**Selector.** `research/experiments/synthetic-mix-ap/selector.py`
