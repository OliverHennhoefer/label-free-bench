"""Example user metric: edit or copy this file when testing a new idea."""

import numpy as np

from label_free_bench.metrics import MetricInput, MetricSpec


def tail_separation(data: MetricInput) -> float:
    """Compare the upper anomaly-score tail with the central score level."""
    scores = np.asarray(data.anomaly_scores)
    cutoff = np.quantile(scores, 1.0 - data.metric_tail_fraction)
    tail = scores[scores >= cutoff]
    scale = np.median(np.abs(scores - np.median(scores))) + 1e-12
    return float((np.mean(tail) - np.median(scores)) / scale)


METRIC = MetricSpec(
    name="user_tail_separation", direction="maximize", compute=tail_separation
)
