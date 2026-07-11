import numpy as np

from label_free_bench.inference import (
    exact_sign_flip_pvalue,
    hierarchical_mean_ci,
    holm_adjust,
)


def test_hierarchical_bootstrap_is_deterministic_and_dataset_equal() -> None:
    rows = [
        {"dataset": "a", "value": 0.0},
        {"dataset": "a", "value": 2.0},
        {"dataset": "b", "value": 10.0},
        {"dataset": "b", "value": 10.0},
    ]

    first = hierarchical_mean_ci(rows, "value", rounds=500, seed=9)
    second = hierarchical_mean_ci(rows, "value", rounds=500, seed=9)

    assert first == second
    assert first[0] == 5.5
    assert first[1] <= first[0] <= first[2]


def test_exact_sign_flip_pvalue_matches_complete_enumeration() -> None:
    assert exact_sign_flip_pvalue([1.0, 1.0]) == 0.5
    assert exact_sign_flip_pvalue([0.0, 0.0]) == 1.0


def test_holm_adjustment_preserves_order_and_monotonicity() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.03])

    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
