# Idan 2024: unsupervised validation by ensemble consensus

**Source.** "Towards Unsupervised Validation of Anomaly-Detection Models", arXiv 2410.14579, DOI 10.3233/FAIA240859. https://arxiv.org/abs/2410.14579

**Establishes.** Builds an "accurately diverse" ensemble whose members agree on the strongest outliers and disagree on the order of ordinary inliers, measured by multi-way rank correlations, and uses it to select and evaluate models. Reported (from a partly summarized PDF, not verified verbatim): Spearman between its divergence score and PR AUC of about 0.84 to 0.96 on roughly ten tabular sets.

**Setting.** Fully unsupervised, contaminated data containing the outliers to be found.

**Relevance.** Its consensus signal is computed on the outliers themselves. Our selection rows hold only normals, so agreement on the top of the score list measures agreement on peripheral normals, which may or may not transfer. Treat the reported correlations as unverified.

**Checked.** 2026-11-23, abstract and truncated PDF via web fetch. Query: "arxiv 2410.14579 unsupervised anomaly detection model selection ensemble consensus".
