# Ding, Wen, Klüttermann et al. 2026: OutFormer

**Source.** "From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection", arXiv 2602.03018 v2, May 2026. https://arxiv.org/abs/2602.03018

**Establishes.** A prior-fitted transformer pretrained on synthetic labeled tabular data (Gaussian-mixture, structural-causal and copula priors, bandit curriculum) scores test rows in one forward pass with inlier training data in context. On ADBench it reports average rank 4.02 against 4.74 for TabPFN-OD and 6.00 for FoMo-0D, win rate 0.71.

**Setting.** Tabular, zero-shot, inductive, inlier-only context data.

**Relevance.** It sidesteps selection rather than solving it, so it is not a selector for our pool. Its priors are a source of synthetic anomaly types (copula and causal perturbations) that a synthetic-validation selector could borrow. Adding it as a candidate would be a protocol change (README Q1).

**Checked.** 2026-11-23, HTML full text via web fetch. Query: "arxiv 2602.03018 OutFormer zero-shot tabular anomaly detection".
