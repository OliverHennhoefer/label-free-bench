# Goswami, Challu, Callot et al. 2023: model selection for time-series anomaly detection

**Source.** "Unsupervised Model Selection for Time-Series Anomaly Detection", arXiv 2210.01078 v3, January 2023. https://arxiv.org/abs/2210.01078

**Establishes.** Three surrogate classes, forecasting error, synthetic anomaly injection and model centrality, are combined by robust Borda aggregation, and a minimum-influence metric down-weights unreliable rankings. Over 19 configurations on UCR subsets and SMD, robust Borda and the minimum-influence variant have the fewest significant losses to the oracle, and the latter is never significantly worse than selection with 20% labels.

**Setting.** Time series, unlabeled; adjusted best F1.

**Relevance.** Supports synthetic injection as a surrogate and suggests aggregation across surrogate types is more robust than any one. Our basic bank's robust Borda aggregates different metrics (ASOI, Laplacian, SIREOS, MC, HITS), not injection.

**Checked.** 2026-11-23, PDF via web fetch. Query: "arxiv 2210.01078" (via AutoUAD search results).
