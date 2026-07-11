from pathlib import Path

import numpy as np
import pytest

from label_free_bench.data import fetch_adbench_data, load_dataset, make_benchmark_split


def test_adbench_numbered_file_is_discovered(tmp_path: Path) -> None:
    X = np.arange(120, dtype=float).reshape(30, 4)
    y = np.array([0] * 24 + [1] * 6)
    np.savez(tmp_path / "6_cardio.npz", X=X, y=y)

    loaded_X, loaded_y = load_dataset("cardio", data_dir=tmp_path)

    np.testing.assert_array_equal(loaded_X, X)
    np.testing.assert_array_equal(loaded_y, y)


def test_stratified_test_preserves_prevalence_and_hides_development_anomalies() -> None:
    X, y = load_dataset("synthetic")
    split = make_benchmark_split(X, y, seed=3, test_fraction=0.5)

    assert len(split.development_normals) == 120
    assert split.test_labels.sum() == 20
    assert len(split.test_features) == 140
    assert abs(split.test_labels.mean() - y.mean()) <= 1 / len(split.test_labels)
    normal_rows = {tuple(row) for row in X[y == 0]}
    assert all(tuple(row) in normal_rows for row in split.development_normals)


def test_normal_sample_cap_is_seeded() -> None:
    X, y = load_dataset("synthetic")
    first = make_benchmark_split(
        X, y, seed=3, test_fraction=0.5, max_normal_samples=100
    )
    second = make_benchmark_split(
        X, y, seed=3, test_fraction=0.5, max_normal_samples=100
    )

    assert len(first.development_normals) == 100
    np.testing.assert_array_equal(first.development_normals, second.development_normals)
    np.testing.assert_array_equal(first.test_features, second.test_features)
    assert first.split_hash == second.split_hash

    changed = make_benchmark_split(
        X, y, seed=4, test_fraction=0.5, max_normal_samples=100
    )
    assert changed.split_hash != first.split_hash


def test_fetch_data_rejects_names_outside_the_versioned_registry(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unknown benchmark"):
        fetch_adbench_data(tmp_path, {"not-a-dataset"})
