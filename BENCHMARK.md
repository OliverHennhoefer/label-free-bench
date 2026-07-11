# Benchmark contract

## Claim

`label-free-bench` measures how well a normal-only metric selects a detector and
hyperparameters from a fixed candidate pool. It does not measure detector quality
in isolation and does not claim that labeled anomalies are unnecessary for final
scientific evaluation.

## V1 protocol

- The complete labeled dataset is stratified 50:50.
- Development anomalies are discarded before preprocessing, fitting, or metric
  evaluation. Development normals are capped at 500.
- Five detector families each receive the same candidate budget. The first
  candidate is the documented PyOD default; the remainder follow the deterministic
  candidate-pool v1.0 Latin-hypercube template.
- For weighted PCA on exactly rank-deficient normal data, zero explained-variance
  weights are floored at the smallest positive component weight. This conservative
  numerical rule prevents PyOD's division by zero without giving a null component
  unbounded influence.
- The first occurrence of each detector is a default anchor. Candidate budgets
  are nested and sensitivity results are published.
- Metrics see only frozen `MetricInput` arrays derived from development normals.
- Source-prevalence Average Precision is primary. ROC-AUC is secondary.
- AP at 1%, 5%, and 10% is computed by class-prior reweighting, holding empirical
  class-conditional score distributions fixed. It is a prior-shift sensitivity
  analysis, not a claim that the datasets literally have those prevalences.
- A metric failure receives normalized regret 1 and remains in every comparison.
  Quality conditional on success and success rate are both reported.

## Primary endpoint

For a fixed candidate pool,

```text
normalized AP regret = (oracle AP - selected AP) / (oracle AP - worst AP)
```

It is zero when all candidates tie. This endpoint is conditional on candidate-pool
version 1.0 and must not be interpreted as an intrinsic property of a metric.
Raw regret, supervised percentile, lift and efficiency relative to expected random
candidate selection, default-detector baselines, and nested candidate budgets are
reported alongside it.

## Inference

Datasets are the independent publication units; seeds quantify split variability.
Point estimates weight datasets equally. Confidence intervals use a deterministic
10,000-round hierarchical bootstrap over datasets and then seeds.

Development-suite results are exploratory: they never receive p-values. A metric
must be fixed before a confirmation run. On the confirmation suite only, the
primary paired dataset-mean effect receives an exact two-sided sign-flip test and
Holm correction across reference comparisons. This test assumes exchangeability
of method labels within dataset under the null; effect size and interval remain
the primary evidence.

The public confirmation labels cannot technically prevent repeated local use.
Repeating confirmation after inspecting its result invalidates the confirmatory
claim and must be disclosed.

## Frozen evidence

A reference run stores checksummed, normal-only evidence per dataset and seed.
Custom metric runs load those arrays and never refit detectors. The manifest binds
the semantic configuration, canonical dataset hashes, split hashes, candidate
pool, evidence hashes, Git state, platform, and exact dependency versions.

## Scope and limitations

V1 evaluates clean-normal, IID tabular one-class learning. It does not cover
contaminated development normals, temporal drift, deep detectors, or thresholded
operational decisions. These are explicit later robustness axes, not hidden claims
of v1.
