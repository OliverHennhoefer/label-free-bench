# label-free-bench v1 plan

## Purpose

Provide a reproducible benchmark for label-free anomaly-detector model selection.
A metric ranks a frozen pool of detector/hyperparameter candidates using normal
development data only. Hidden labels evaluate the selection afterward.

## V1 research contract

1. Download and checksum the 15 canonical ADBench datasets.
2. Use the fixed seven-dataset development suite for human or agent iteration.
3. Stratify each complete dataset 50:50, discard development anomalies, cap the
   remaining normal pool at 500, and fit preprocessing on development data only.
4. Evaluate the balanced candidate-pool v1.1 design: five PyOD families, equal
   budgets, deterministic stratified templates, unique PCA settings, and default
   anchors.
5. Freeze normal-only validation features, scores, repeated bootstrap-fit scores,
   reference scores, probabilities, and support information per dataset and seed.
6. Evaluate every metric against exactly that evidence. Reuse it for new metrics;
   never refit detectors during iterative comparison.
7. Use source-prevalence Average Precision regret as the primary selection
   endpoint. Report raw regret, random-selection efficiency, percentile, ROC-AUC,
   rank agreement, default/random baselines, failures, runtime, candidate-budget
   sensitivity, and domain sensitivity.
8. Evaluate 1%, 5%, and 10% prior-shift robustness by deterministic AP class
   weighting rather than stochastic subsampling.
9. Treat failures as worst-case selection outcomes and report success separately.
10. Use dataset-equal hierarchical confidence intervals. Label development results
    exploratory; reserve corrected primary testing for one fixed confirmation run.

## User metric contract

A `MetricSpec` under `src/label_free_bench/user_metrics/` consumes only
`MetricInput`, returns one finite deterministic scalar per candidate, and declares
whether to maximize or minimize it. Metric code must not access labels, test data,
prevalence, dataset loaders, or result artifacts.

## Release artifacts

- One ranked `results.csv` for normal use.
- Candidate, selection, dataset, domain, budget, comparison, and runtime tables.
- Per-split frozen evidence and true resume caches.
- A manifest binding benchmark version, candidate pool, semantic config, canonical
  dataset hashes, split/evidence hashes, Git state, platform, and dependencies.
- Checksummed downloader, dataset registry/cards, CI, contribution instructions,
  citation metadata, and compact published reference tables.

## Beyond v1

Later versions may add contaminated normal pools, temporal or covariate shift,
more detector families, adaptive metric-driven search as a separate track, and an
externally hosted confirmation leaderboard. These extensions must not alter the
frozen v1 contract.
