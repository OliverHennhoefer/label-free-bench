# label-free-bench

`label-free-bench` evaluates metrics that select a tabular anomaly detector and
hyperparameters using normal data only. Hidden anomaly labels are used afterward
to measure whether the selection was good; they never enter metric inputs.

The benchmark accompanies
[`labelfree`](https://github.com/OliverHennhoefer/label-free). V1 uses 15
checksummed classical [ADBench](https://github.com/Minqi824/ADBench) datasets and
five complementary PyOD families: Isolation Forest, LOF, PCA, One-Class SVM, and
HBOS.

## Install and fetch data

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen
uv run label-free-bench fetch-data
uv run label-free-bench validate --config configs/benchmark.yaml
```

## Run the reference benchmark

The light reference is the normal iteration target. It uses the seven
development datasets, one split seed, ten balanced candidates, and every
reference metric:

```bash
uv run label-free-bench run \
  --config configs/light.yaml \
  --output results/reference-light
```

The publication development reference uses 60 candidates—12 per detector—and
ten seeds:

```bash
uv run label-free-bench run \
  --config configs/benchmark.yaml \
  --suite development \
  --output results/reference-development
```

Completed dataset/seed splits are checksummed and resumed without refitting.
Isolation Forest and LOF use available CPU cores; other work remains deterministic
and split-sequential.

## Evaluate a new metric without refitting

Put the metric under `src/label_free_bench/user_metrics/`:

```python
from label_free_bench.metrics import MetricInput, MetricSpec


def my_score(data: MetricInput) -> float:
    return float(data.anomaly_scores.std())


METRIC = MetricSpec("my_score", "maximize", my_score)
```

Then evaluate it directly from the frozen normal-only reference evidence:

```bash
uv run label-free-bench run \
  --config configs/light.yaml \
  --metric label_free_bench.user_metrics.my_score:METRIC \
  --output results/my-score \
  --reference results/reference-light
```

No detector is fitted in this command. Evidence and candidate hashes are checked
before evaluation.

## Confirmation

Iterative results are exploratory. Fix the metric before using the eight-dataset
confirmation suite, then run it once:

```bash
uv run label-free-bench run \
  --config configs/benchmark.yaml \
  --suite confirmation \
  --confirm \
  --output results/reference-confirmation
```

Evaluate the fixed metric against that reference with `--reference`. Only a
confirmation comparison receives the exact primary sign-flip p-value and Holm
correction. Reusing confirmation after seeing its result invalidates that claim.

## Results

Every output has one human-facing file:

```text
results/<run>/
├── results.csv
└── artifacts/
```

Reference `results.csv` is a ranked metric leaderboard. A custom run contains
only one row per submitted metric and prevalence condition, with its rank among
the frozen references, number of references beaten, delta to the best and random
references, and counts of corrected significant wins/losses. Full pairwise detail
remains under `artifacts/`.

The clean file also contains:

- benchmark, candidate-pool, suite, and inference identity;
- dataset/split coverage and overall rank;
- normalized-regret selection quality and confidence interval;
- raw Average Precision, random-selection efficiency, ROC-AUC, near-oracle,
  oracle-hit, Spearman, and Kendall effects with intervals;
- success/failure and metric-runtime effects;
- selected detector distribution and dataset win/tie/loss counts; and
- p-values only for the preregistered confirmation primary endpoint.

Higher effect values are consistently better. For standalone rows,
`selection_quality_effect = 1 - normalized regret`; for comparisons it is the
reference-to-candidate regret reduction.

`artifacts/` contains candidate and selection tables, frozen evidence, manifests,
per-dataset/domain summaries, nested candidate-budget sensitivity, and raw paired
comparisons.

Source prevalence is primary. The 1%, 5%, and 10% rows use deterministic
class-prior reweighting of the full labeled test half, not anomaly duplication or
repeated subsampling.

## Scientific scope

Read [BENCHMARK.md](BENCHMARK.md) for the estimand, inference assumptions, failure
policy, and limitations. [DATASETS.md](DATASETS.md) documents inclusion criteria,
checksums, and the fixed development/confirmation split.

V1 is a clean-normal, IID tabular one-class benchmark. It does not yet claim
robustness to contaminated development data, temporal drift, or operational
threshold selection.

## Development and contribution

```bash
uv run ruff check .
uv run pytest
uv lock --check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the metric submission contract and
[PLAN.md](PLAN.md) for project rationale. The versioned baseline tables are in
[REFERENCE_RESULTS.md](REFERENCE_RESULTS.md).
