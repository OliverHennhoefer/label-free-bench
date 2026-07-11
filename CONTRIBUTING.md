# Contributing metrics

Implement metrics only under `src/label_free_bench/user_metrics/`. A submission
must return one deterministic finite scalar per candidate, declare its direction,
and use only `MetricInput`. Labels, test data, prevalence, dataset loaders, and
result artifacts are forbidden inputs.

Before submitting:

```bash
uv run pytest
uv run ruff check .
uv run label-free-bench run \
  --config configs/light.yaml \
  --metric label_free_bench.user_metrics.<module>:METRIC \
  --output results/my-metric \
  --reference results/reference-light
```

Include the formula, assumptions, direction, expected failure modes, computational
complexity, source citations, and the exploratory `results.csv`. Do not present a
confirmation result unless the metric was fixed beforehand and confirmation was
run only once.
