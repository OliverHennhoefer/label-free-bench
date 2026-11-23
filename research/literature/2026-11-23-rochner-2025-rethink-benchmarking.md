# Röchner, Rothlauf, Klüttermann 2025: rethinking anomaly-detection benchmarks

**Source.** "We Need to Rethink Benchmarking in Anomaly Detection", arXiv 2507.15584, position paper. https://arxiv.org/abs/2507.15584

**Establishes.** A "Quantiles" baseline scores a row by the fraction of its features outside the training 5% and 95% quantiles. In a critical-difference diagram on ADBench ROC AUC with normal-only training, it ranks second, behind only DTE and ahead of DeepSVDD, GoAD and OCSVM; the gap to DTE is not significant. It cannot detect anomalies inside a normal annulus.

**Setting.** Tabular ADBench, normal-only training, standard hyperparameters.

**Relevance.** Many ADBench anomalies are marginal (per-feature) extremes. A selector whose synthetic anomalies are only marginal extremes will favour detectors like HBOS; one that also breaks feature dependence probes joint structure. This bears on what kind of synthetic anomaly predicts real ADBench outcomes.

**Checked.** 2026-11-23, HTML v1 via web fetch. Query: "arxiv 2507.15584 Röchner per-feature quantile baseline anomaly detection ADBench".
