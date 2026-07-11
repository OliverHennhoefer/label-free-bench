from dataclasses import fields

import numpy as np

from label_free_bench.metrics import MetricInput, load_metrics


def _input() -> MetricInput:
    scores = np.linspace(0.1, 1.0, 10)
    return MetricInput(
        validation_features=np.arange(20, dtype=float).reshape(10, 2),
        anomaly_scores=scores,
        metric_tail_fraction=0.2,
        reference_scores=np.linspace(0.0, 1.0, 12),
        support_volume=20.0,
        repeated_anomaly_scores=np.vstack((scores, scores)),
        outlier_probabilities=scores,
    )


def test_metric_input_has_no_label_or_test_field() -> None:
    names = {field.name for field in fields(MetricInput)}

    assert names == {
        "validation_features",
        "anomaly_scores",
        "metric_tail_fraction",
        "reference_scores",
        "support_volume",
        "repeated_anomaly_scores",
        "outlier_probabilities",
    }


def test_user_metric_loads_and_returns_finite_value() -> None:
    (metric,) = load_metrics(("label_free_bench.user_metrics.metric:METRIC",))
    value = metric.evaluate((_input(),))[0]

    assert metric.name == "user_tail_separation"
    assert np.isfinite(value)


def test_reference_collection_contains_every_scalar_selection_metric() -> None:
    metrics = load_metrics(("label_free_bench.metrics:REFERENCE_METRICS",))

    assert {metric.name for metric in metrics} == {
        "score_silhouette",
        "score_calinski_harabasz",
        "score_davies_bouldin",
        "asi",
        "asoi",
        "rtm",
        "eag",
        "npd",
        "laplacian",
        "model_centrality",
        "average_rank_consensus",
        "hits",
        "ranking_stability",
        "top_k_stability",
        "ireos",
        "sireos",
        "excess_mass",
        "mass_volume",
    }
