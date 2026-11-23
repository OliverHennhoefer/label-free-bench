# Prediction: cross-family-tail-consensus

**Mechanism (written first,).** Selection rows are all normal, so the only anomaly-relevant structure in their scores is the periphery: the rows a detector puts in its top 5%. If several detector families with different inductive biases agree that a row is peripheral, it most likely lies in genuinely low density, where real anomalies lie beyond. A candidate whose top 5% matches the periphery agreed by the other families tracks density better. Rank each candidate by the overlap of its top-5% set with the top 5% of a consensus built from the mean rank of each other family (one vote per family, own family excluded), using fractional membership at ties. No refits.

**Discriminating prediction.** Target `ap`, reference `mc`, development bank 65e006afd99c. Model centrality (`mc`, 0.413 in run a1330c55e644) uses agreement over the whole ranking with every candidate, so same-family duplicates vote for each other. The mechanism predicts a mean AP at least 0.02 above `mc` with source wins on more than half the sources. It is wrong if it does not beat `mc`, which would show that excluding own-family votes and focusing on the tail adds nothing over plain centrality.

**Prior entries.** No claims or negative rows exist yet; nothing to adopt or reject.

**Literature (2026-11-23).** Query: "outlier ensemble consensus top-k agreement across detector families model selection unsupervised". Idan 2024 uses agreement on strong outliers in contaminated data; Ma 2021 found consensus no better than random-hyperparameter Isolation Forest. SELECT (arXiv 1501.01924) is queued.

**Selector.** `research/experiments/cross-family-tail-consensus/selector.py`
