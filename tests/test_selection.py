import numpy as np
import pytest

from label_free_bench.metrics import MetricSpec
from label_free_bench.pipeline import _select, _supervised_scores


def test_supervised_scores_match_hand_computed_example() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    average_precision, roc_auc, prevalence = _supervised_scores(labels, scores)

    assert average_precision == pytest.approx(5 / 6)
    assert roc_auc == pytest.approx(0.75)
    assert prevalence == 0.5


def _rows(average_precision: list[float]) -> list[dict[str, object]]:
    metric_values = [0.1, 0.3, 0.2]
    return [
        {
            "candidate_id": index,
            "detector": f"model-{index}",
            "test_average_precision": value,
            "test_roc_auc": [0.4, 0.6, 0.9][index],
            "metric_test": metric_values[index],
        }
        for index, value in enumerate(average_precision)
    ]


def test_selection_metrics_match_hand_computed_example() -> None:
    metric = MetricSpec("test", "maximize", compute=lambda data: 0.0)

    (selected,) = _select(_rows([0.2, 0.5, 0.8]), (metric,), "dataset", 1)

    assert selected["selected_average_precision"] == 0.5
    assert selected["oracle_average_precision"] == 0.8
    assert selected["worst_average_precision"] == 0.2
    assert selected["average_precision_regret"] == pytest.approx(0.3)
    assert selected["normalized_average_precision_regret"] == pytest.approx(0.5)
    assert selected["average_precision_lift_over_mean"] == pytest.approx(0.0)
    assert selected["selected_supervised_percentile"] == pytest.approx(0.5)
    assert selected["near_oracle"] is False
    assert selected["oracle_hit"] is False
    assert selected["selected_roc_auc"] == 0.6
    assert selected["spearman_average_precision"] == pytest.approx(0.5)
    assert selected["kendall_average_precision"] == pytest.approx(1 / 3)


def test_tied_candidate_quality_has_zero_regret_and_all_are_oracles() -> None:
    metric = MetricSpec("test", "maximize", compute=lambda data: 0.0)

    (selected,) = _select(_rows([0.5, 0.5, 0.5]), (metric,), "dataset", 1)

    assert selected["normalized_average_precision_regret"] == 0.0
    assert selected["selected_supervised_percentile"] == 1.0
    assert selected["near_oracle"] is True
    assert selected["oracle_hit"] is True


def test_metric_failure_is_retained_as_worst_selection() -> None:
    metric = MetricSpec("test", "maximize", compute=lambda data: 0.0)
    rows = _rows([0.2, 0.5, 0.8])
    for row in rows:
        row["metric_test"] = ""
        row["error_test"] = "RuntimeError: failed"

    (selected,) = _select(rows, (metric,), "dataset", 1)

    assert selected["status"] == "failed"
    assert selected["success"] is False
    assert selected["normalized_average_precision_regret"] == 1.0
    assert selected["selected_average_precision"] == 0.2
    assert selected["failure_reason"] == "RuntimeError: failed"
