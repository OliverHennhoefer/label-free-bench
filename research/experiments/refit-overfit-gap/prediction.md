# Prediction: refit-overfit-gap

**Mechanism (written first,).** At a fixed 5% alert budget, AP on the private test is hurt by false alarms among unseen normals. A detector that fits its training normals much more tightly than new normals (an overfit density boundary, such as small-neighbourhood LOF on few rows) raises the scores of unseen normals and so ranks more of them above anomalies. Measure the gap directly: refit each candidate on `views["train"]` by matrix position (verified against `batch.scores` as in synthetic-mix-ap), take its training scores, and compute the AUC of selection-normal scores against training-normal scores. Rank candidates by the negative of |AUC − 0.5|, smaller gap preferred, ties broken by the default order.

**Discriminating prediction.** Target `ap`, reference `fixed/KNN/5`, development bank 65e006afd99c. The mechanism predicts a mean AP above `uniform_model` (0.399, run a1330c55e644) and within 0.03 of KNN/5. It is wrong if the selector scores below default Isolation Forest (0.376) or picks HBOS on more than half the tasks: that would show the gap rewards coarse, insensitive scores rather than well-generalizing ones.

**Prior entries.** No claims or negative rows exist yet; nothing to adopt or reject.

**Literature (2026-11-23).** Queries: "arxiv 2606.08742 AUCp pseudo-AUC"; "outlier detection model selection overfitting train vs held-out normal scores generalization gap one-class". AUCp (Siddiquee 2026) contrasts training normals with unlabeled test data; here no test rows exist, so the contrast becomes a generalization gap. "Friend or Foe? Harnessing Controllable Overfitting for Anomaly Detection" (arXiv 2412.00560) is unverified and queued.

**Selector.** `research/experiments/refit-overfit-gap/selector.py`
