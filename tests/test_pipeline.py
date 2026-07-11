import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import label_free_bench.pipeline as pipeline_module
from label_free_bench.config import DatasetConfig, load_config
from label_free_bench.pipeline import (
    _prior_adjusted_average_precision,
    compare_runs,
    evaluate_metrics_from_reference,
    run_benchmark,
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_smoke_run_is_reproducible_and_writes_audit_tables(tmp_path: Path) -> None:
    config = replace(
        load_config("configs/light.yaml"),
        datasets=(DatasetConfig("synthetic"),),
        seeds=(7,),
        trials=5,
        metrics=(
            "label_free_bench.metrics:EAG",
            "label_free_bench.metrics:LAPLACIAN",
            "label_free_bench.user_metrics.metric:METRIC",
        ),
    )
    first = run_benchmark(config, tmp_path / "first")
    second = run_benchmark(config, tmp_path / "second")

    first_candidates = _read_rows(first.candidates_path)
    second_candidates = _read_rows(second.candidates_path)
    stable_columns = [
        "dataset",
        "seed",
        "candidate_id",
        "detector",
        "params",
        "test_roc_auc",
        "test_average_precision",
        "test_anomaly_prevalence",
        "metric_eag",
        "metric_laplacian",
        "metric_user_tail_separation",
    ]
    assert len(first_candidates) == 5
    assert [{key: row[key] for key in stable_columns} for row in first_candidates] == [
        {key: row[key] for key in stable_columns} for row in second_candidates
    ]

    selections = _read_rows(first.selections_path)
    assert {row["metric"] for row in selections} == {
        "eag",
        "laplacian",
        "user_tail_separation",
        "random_candidate",
        "default_iforest",
        "default_lof",
        "default_pca",
        "default_ocsvm",
        "default_hbos",
    }
    assert all(
        0 <= float(row["normalized_average_precision_regret"]) <= 1
        for row in selections
    )
    assert first.dataset_summary_path.exists()
    assert first.summary_path.exists()
    manifest = (first.output_dir / "artifacts" / "manifest.json").read_text()
    assert '"result_schema_version": 5' in manifest
    assert (first.output_dir / "artifacts" / "evidence").is_dir()
    assert (first.output_dir / "artifacts" / "splits").is_dir()
    parsed_manifest = json.loads(manifest)
    assert parsed_manifest["candidate_pool_version"] == "1.1"
    assert parsed_manifest["dataset_hashes"]["synthetic"]
    assert parsed_manifest["splits"]["synthetic:seed=7"]["split_hash"]
    assert parsed_manifest["environment"]["packages"]["labelfree"]
    evidence_path = first.output_dir / parsed_manifest["splits"][
        "synthetic:seed=7"
    ]["evidence_path"]
    with np.load(evidence_path, allow_pickle=False) as evidence:
        assert set(evidence.files) == {
            "validation_features",
            "anomaly_scores",
            "reference_scores",
            "repeated_anomaly_scores",
            "outlier_probabilities",
            "support_volume",
            "metric_tail_fraction",
        }
    assert first.results_path.exists()
    assert {path.name for path in first.output_dir.iterdir()} == {
        "results.csv",
        "artifacts",
    }
    clean_results = _read_rows(first.results_path)
    assert {row["condition"] for row in clean_results} == {
        "original",
        "controlled",
    }
    assert all(row["result_kind"] == "absolute" for row in clean_results)
    assert first.prevalence_sensitivity_path.exists()
    assert first.prevalence_summary_path.exists()
    prevalence_rows = _read_rows(first.prevalence_sensitivity_path)
    second_prevalence_rows = _read_rows(second.prevalence_sensitivity_path)
    assert [
        {key: value for key, value in row.items() if "seconds" not in key}
        for row in prevalence_rows
    ] == [
        {key: value for key, value in row.items() if "seconds" not in key}
        for row in second_prevalence_rows
    ]
    assert {float(row["target_anomaly_prevalence"]) for row in prevalence_rows} == {
        0.01,
        0.05,
        0.1,
    }
    assert all(
        row["prevalence_method"] == "prior_reweighting"
        for row in prevalence_rows
    )


def test_all_references_run_once_and_custom_metric_compares_to_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = run_benchmark(
        replace(
            load_config("configs/light.yaml"),
            datasets=(DatasetConfig("synthetic"),),
            seeds=(7,),
            trials=5,
        ),
        tmp_path / "reference",
    )
    reference_selections = _read_rows(reference.selections_path)

    assert len(reference_selections) == 24
    assert all(row["status"] == "ok" for row in reference_selections)

    def fail_if_fitted(*args: object, **kwargs: object) -> None:
        raise AssertionError("frozen metric evaluation attempted to fit a detector")

    monkeypatch.setattr(pipeline_module, "build_detector", fail_if_fitted)
    candidate = evaluate_metrics_from_reference(
        replace(
            load_config("configs/light.yaml"),
            datasets=(DatasetConfig("synthetic"),),
            seeds=(7,),
            trials=5,
            metrics=("label_free_bench.user_metrics.metric:METRIC",),
        ),
        reference.output_dir,
        tmp_path / "candidate",
    )
    comparison_path = compare_runs(reference.output_dir, candidate.output_dir)
    results = _read_rows(comparison_path)
    comparisons = _read_rows(candidate.output_dir / "artifacts" / "comparison.csv")

    assert len(comparisons) == 24
    assert {row["candidate_metric"] for row in comparisons} == {"user_tail_separation"}
    assert {row["reference_metric"] for row in comparisons} == {
        row["metric"] for row in reference_selections
    }
    assert all(int(row["dataset_count"]) == 1 for row in comparisons)
    assert all("primary_p_value_holm" in row for row in comparisons)
    assert all(row["inference_mode"] == "exploratory" for row in comparisons)
    assert all(math.isnan(float(row["primary_p_value"])) for row in comparisons)
    assert all("normalized_regret_reduction_ci_low" in row for row in comparisons)
    assert all("average_precision_delta_ci_high" in row for row in comparisons)
    assert all("regret_reduction" not in row for row in comparisons)
    assert len(results) == 4
    assert all(row["result_kind"] == "candidate" for row in results)
    assert all(row["selection_quality_effect"] != "" for row in results)
    assert {row["condition"] for row in results} == {"original", "controlled"}
    assert all(int(row["reference_count"]) == 24 for row in results)
    assert (candidate.output_dir / "artifacts" / "comparison_dataset.csv").exists()
    controlled_comparisons = _read_rows(
        candidate.output_dir / "artifacts" / "comparison_prevalence.csv"
    )
    assert len(controlled_comparisons) == 24 * 3
    assert all("primary_p_value" not in row for row in controlled_comparisons)

    manifest_path = candidate.output_dir / "artifacts" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("result_schema_version")
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="regenerate"):
        compare_runs(reference.output_dir, candidate.output_dir)


def test_prior_adjusted_average_precision_uses_all_samples_and_target_prior() -> None:
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.35, 0.8])

    balanced = _prior_adjusted_average_precision(labels, scores, 0.5)
    rare = _prior_adjusted_average_precision(labels, scores, 0.1)

    assert balanced == pytest.approx(5 / 6)
    assert 0.1 <= rare < balanced


def test_completed_split_is_resumed_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(
        load_config("configs/light.yaml"),
        datasets=(DatasetConfig("synthetic"),),
        seeds=(3,),
        trials=5,
        metrics=("label_free_bench.metrics:EAG",),
    )
    output = tmp_path / "resume"
    first = run_benchmark(config, output)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("detectors were refitted instead of resuming")

    monkeypatch.setattr(pipeline_module, "_evaluate_candidates", fail_if_called)
    second = run_benchmark(config, output)

    assert _read_rows(first.candidates_path) == _read_rows(second.candidates_path)


def test_only_confirmation_comparison_receives_primary_p_values(
    tmp_path: Path,
) -> None:
    base = replace(
        load_config("configs/light.yaml"),
        datasets=(DatasetConfig("synthetic", suite="confirmation"),),
        seeds=(2,),
        trials=5,
        metrics=("label_free_bench.metrics:EAG",),
    )
    reference = run_benchmark(
        base,
        tmp_path / "confirmation-reference",
        suite="confirmation",
        inference_mode="confirmatory",
    )
    candidate = evaluate_metrics_from_reference(
        replace(
            base,
            metrics=("label_free_bench.user_metrics.metric:METRIC",),
        ),
        reference.output_dir,
        tmp_path / "confirmation-candidate",
    )

    compare_runs(reference.output_dir, candidate.output_dir)
    comparisons = _read_rows(candidate.output_dir / "artifacts" / "comparison.csv")

    assert all(row["inference_mode"] == "confirmatory" for row in comparisons)
    assert all(math.isfinite(float(row["primary_p_value"])) for row in comparisons)
