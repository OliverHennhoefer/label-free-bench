"""End-to-end controlled candidate and metric benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from labelfree.metrics import bounding_box_volume
from scipy.stats import kendalltau, rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from label_free_bench import __version__
from label_free_bench.config import RunConfig, config_dict, experiment_dict
from label_free_bench.data import (
    dataset_hash,
    dataset_registry,
    load_dataset,
    make_benchmark_split,
    split_development_normals,
)
from label_free_bench.inference import (
    exact_sign_flip_pvalue,
    hierarchical_mean_ci,
    holm_adjust,
    stable_seed,
)
from label_free_bench.metrics import MetricInput, MetricSpec, load_metrics
from label_free_bench.models import (
    CANDIDATE_POOL_VERSION,
    CandidateSpec,
    build_detector,
    candidate_specs,
    fit_detector,
)

RESULT_SCHEMA_VERSION = 5
BOOTSTRAP_ROUNDS = 10_000


@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    results_path: Path
    candidates_path: Path
    selections_path: Path
    dataset_summary_path: Path
    summary_path: Path
    prevalence_sensitivity_path: Path
    prevalence_summary_path: Path


@dataclass(frozen=True)
class EvidenceBundle:
    """Frozen normal-only evidence from one dataset/seed candidate pool."""

    validation_features: np.ndarray
    anomaly_scores: np.ndarray
    reference_scores: np.ndarray
    repeated_anomaly_scores: np.ndarray
    outlier_probabilities: np.ndarray
    support_volume: float
    metric_tail_fraction: float

    def metric_inputs(self) -> list[MetricInput]:
        return [
            MetricInput(
                validation_features=self.validation_features,
                anomaly_scores=self.anomaly_scores[index],
                metric_tail_fraction=self.metric_tail_fraction,
                reference_scores=self.reference_scores[index],
                support_volume=self.support_volume,
                repeated_anomaly_scores=self.repeated_anomaly_scores[index],
                outlier_probabilities=self.outlier_probabilities[index],
            )
            for index in range(len(self.anomaly_scores))
        ]


def compare_runs(
    reference_dir: str | Path,
    candidate_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Compare new metrics with stored references using paired split effects."""
    reference = Path(reference_dir)
    candidate = Path(candidate_dir)
    reference_artifacts = reference / "artifacts"
    candidate_artifacts = candidate / "artifacts"
    reference_absolute_rows = _read_csv(reference / "results.csv")
    candidate_absolute_rows = _read_csv(candidate / "results.csv")
    reference_manifest = json.loads((reference_artifacts / "manifest.json").read_text())
    candidate_manifest = json.loads((candidate_artifacts / "manifest.json").read_text())
    if (
        reference_manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION
        or candidate_manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION
    ):
        raise ValueError(
            "result schema is incompatible; regenerate both reference and "
            "candidate runs"
        )
    if reference_manifest.get("experiment_hash") != candidate_manifest.get(
        "experiment_hash"
    ):
        raise ValueError(
            "reference and candidate runs use different datasets, seeds, models, "
            "trial budgets, or evidence settings"
        )
    expected_reference_hash = candidate_manifest.get("reference_manifest_hash")
    if expected_reference_hash and expected_reference_hash != _sha256_file(
        reference_artifacts / "manifest.json"
    ):
        raise ValueError("candidate run was evaluated from a different reference")
    inference_mode = str(candidate_manifest.get("inference_mode", "exploratory"))
    confirmatory = (
        inference_mode == "confirmatory"
        and candidate_manifest.get("suite") == "confirmation"
        and reference_manifest.get("inference_mode") == "confirmatory"
    )

    identity_fields = ("dataset", "seed", "candidate_id", "detector", "params")
    reference_candidates = {
        tuple(row[field] for field in identity_fields)
        for row in _read_csv(reference_artifacts / "candidates.csv")
    }
    candidate_candidates = {
        tuple(row[field] for field in identity_fields)
        for row in _read_csv(candidate_artifacts / "candidates.csv")
    }
    if reference_candidates != candidate_candidates:
        raise ValueError("reference and candidate runs contain different candidates")

    reference_rows = _read_csv(reference_artifacts / "selections.csv")
    candidate_rows = _read_csv(candidate_artifacts / "selections.csv")
    comparisons: list[dict[str, Any]] = []
    comparison_datasets: list[dict[str, Any]] = []
    candidate_metrics = sorted({row["metric"] for row in candidate_rows})
    reference_metrics = sorted({row["metric"] for row in reference_rows})
    effect_fields = (
        "normalized_regret_reduction",
        "selection_efficiency_delta",
        "average_precision_delta",
        "roc_auc_delta",
        "spearman_average_precision_delta",
        "kendall_average_precision_delta",
        "near_oracle_rate_delta",
        "oracle_hit_rate_delta",
        "success_rate_delta",
        "runtime_reduction_seconds",
    )

    for candidate_metric in candidate_metrics:
        for reference_metric in reference_metrics:
            if candidate_metric == reference_metric:
                continue
            paired = _paired_effects(
                candidate_rows,
                reference_rows,
                candidate_metric,
                reference_metric,
            )
            if not paired:
                continue
            dataset_rows = _comparison_dataset_rows(
                paired, candidate_metric, reference_metric, effect_fields
            )
            comparison_datasets.extend(dataset_rows)
            result: dict[str, Any] = {
                "candidate_metric": candidate_metric,
                "reference_metric": reference_metric,
                "dataset_count": len(dataset_rows),
                "paired_split_count": len(paired),
            }
            primary_dataset_effects = np.asarray(
                [row["normalized_regret_reduction"] for row in dataset_rows],
                dtype=float,
            )
            result["dataset_wins"] = int(np.sum(primary_dataset_effects > 1e-15))
            result["dataset_ties"] = int(
                np.sum(np.isclose(primary_dataset_effects, 0.0, atol=1e-15))
            )
            result["dataset_losses"] = int(
                np.sum(primary_dataset_effects < -1e-15)
            )
            for effect in effect_fields:
                mean, low, high = hierarchical_mean_ci(
                    paired,
                    effect,
                    rounds=BOOTSTRAP_ROUNDS,
                    seed=stable_seed(candidate_metric, reference_metric, effect),
                )
                result[effect] = mean
                result[f"{effect}_ci_low"] = low
                result[f"{effect}_ci_high"] = high
            result["inference_mode"] = inference_mode
            result["primary_p_value"] = (
                exact_sign_flip_pvalue(
                    [row["normalized_regret_reduction"] for row in dataset_rows]
                )
                if confirmatory
                else float("nan")
            )
            comparisons.append(result)

    adjusted = holm_adjust([row["primary_p_value"] for row in comparisons])
    for row, adjusted_value in zip(comparisons, adjusted, strict=True):
        row["primary_p_value_holm"] = float(adjusted_value)
    destination = Path(output_path) if output_path else candidate / "results.csv"
    _write_csv(candidate_artifacts / "comparison.csv", comparisons)
    _write_csv(
        candidate_artifacts / "comparison_dataset.csv", comparison_datasets
    )
    prevalence_comparisons = _compare_prevalence_sensitivity(
        reference_artifacts,
        candidate_artifacts,
        candidate_metrics,
        reference_metrics,
    )
    if prevalence_comparisons:
        _write_csv(
            candidate_artifacts / "comparison_prevalence.csv",
            prevalence_comparisons,
        )
    _write_csv(
        destination,
        _candidate_result_rows(
            candidate_absolute_rows,
            reference_absolute_rows,
            comparisons,
            prevalence_comparisons,
        ),
    )
    return destination


def _candidate_result_rows(
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    original_comparisons: list[dict[str, Any]],
    controlled_comparisons: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create four concise candidate rows; keep pairwise detail in artifacts."""
    output: list[dict[str, Any]] = []
    for source in candidate_rows:
        row: dict[str, Any] = dict(source)
        matching = [
            reference
            for reference in reference_rows
            if reference["condition"] == source["condition"]
            and reference["anomaly_prevalence"] == source["anomaly_prevalence"]
        ]
        candidate_quality = _numeric(source["selection_quality_effect"])
        qualities = [
            _numeric(reference["selection_quality_effect"])
            for reference in matching
        ]
        all_qualities = np.asarray([candidate_quality, *qualities], dtype=float)
        row["result_kind"] = "candidate"
        row["overall_rank"] = float(
            rankdata(-all_qualities, method="average")[0]
        )
        row["reference_count"] = len(matching)
        row["references_beaten"] = sum(
            candidate_quality > quality + 1e-15 for quality in qualities
        )
        if matching:
            best = max(
                matching,
                key=lambda reference: _numeric(
                    reference["selection_quality_effect"]
                ),
            )
            row["best_reference_metric"] = best["metric"]
            best_reference_metric = best["metric"]
            row["delta_to_best_reference"] = candidate_quality - _numeric(
                best["selection_quality_effect"]
            )
        else:
            best_reference_metric = ""
        random_row = next(
            (
                reference
                for reference in matching
                if reference["metric"] == "random_candidate"
            ),
            None,
        )
        row["delta_to_random_candidate"] = (
            ""
            if random_row is None
            else candidate_quality - _numeric(random_row["selection_quality_effect"])
        )
        comparisons = (
            original_comparisons
            if source["condition"] == "original"
            else [
                result
                for result in controlled_comparisons
                if str(result["target_anomaly_prevalence"])
                == source["anomaly_prevalence"]
            ]
        )
        relevant = [
            result
            for result in comparisons
            if result["candidate_metric"] == source["metric"]
        ]
        best_comparison = next(
            (
                result
                for result in relevant
                if result["reference_metric"] == best_reference_metric
            ),
            None,
        )
        if best_comparison is not None:
            row["dataset_wins"] = best_comparison["dataset_wins"]
            row["dataset_ties"] = best_comparison["dataset_ties"]
            row["dataset_losses"] = best_comparison["dataset_losses"]
        row["significant_reference_wins"] = sum(
            _numeric(result.get("primary_p_value_holm", float("nan"))) < 0.05
            and _numeric(result["normalized_regret_reduction"]) > 0
            for result in relevant
        )
        row["significant_reference_losses"] = sum(
            _numeric(result.get("primary_p_value_holm", float("nan"))) < 0.05
            and _numeric(result["normalized_regret_reduction"]) < 0
            for result in relevant
        )
        output.append(row)
    return sorted(
        output,
        key=lambda row: (
            row["condition"] != "original",
            float(row["overall_rank"]),
            str(row["metric"]),
        ),
    )


def _comparison_results_rows(
    original: list[dict[str, Any]],
    controlled: list[dict[str, Any]] | None = None,
    *,
    suite: str = "",
) -> list[dict[str, Any]]:
    """Create the single concise, higher-is-better comparison table."""
    output: list[dict[str, Any]] = []
    for row in original:
        output.append(
            _clean_result_row(
                result_kind="comparison",
                metric=row["candidate_metric"],
                reference_metric=row["reference_metric"],
                condition="original",
                anomaly_prevalence="source",
                dataset_count=row["dataset_count"],
                split_count=row["paired_split_count"],
                selection=row["normalized_regret_reduction"],
                selection_low=row["normalized_regret_reduction_ci_low"],
                selection_high=row["normalized_regret_reduction_ci_high"],
                average_precision=row["average_precision_delta"],
                average_precision_low=row["average_precision_delta_ci_low"],
                average_precision_high=row["average_precision_delta_ci_high"],
                selection_efficiency=row["selection_efficiency_delta"],
                selection_efficiency_low=row["selection_efficiency_delta_ci_low"],
                selection_efficiency_high=row[
                    "selection_efficiency_delta_ci_high"
                ],
                roc_auc=row["roc_auc_delta"],
                roc_auc_low=row["roc_auc_delta_ci_low"],
                roc_auc_high=row["roc_auc_delta_ci_high"],
                near_oracle=row["near_oracle_rate_delta"],
                near_oracle_low=row["near_oracle_rate_delta_ci_low"],
                near_oracle_high=row["near_oracle_rate_delta_ci_high"],
                oracle_hit=row["oracle_hit_rate_delta"],
                oracle_hit_low=row["oracle_hit_rate_delta_ci_low"],
                oracle_hit_high=row["oracle_hit_rate_delta_ci_high"],
                spearman=row["spearman_average_precision_delta"],
                spearman_low=row["spearman_average_precision_delta_ci_low"],
                spearman_high=row["spearman_average_precision_delta_ci_high"],
                kendall=row["kendall_average_precision_delta"],
                kendall_low=row["kendall_average_precision_delta_ci_low"],
                kendall_high=row["kendall_average_precision_delta_ci_high"],
                success_rate=row["success_rate_delta"],
                success_rate_low=row["success_rate_delta_ci_low"],
                success_rate_high=row["success_rate_delta_ci_high"],
                runtime_seconds=row["runtime_reduction_seconds"],
                runtime_seconds_low=row["runtime_reduction_seconds_ci_low"],
                runtime_seconds_high=row["runtime_reduction_seconds_ci_high"],
                dataset_wins=row["dataset_wins"],
                dataset_ties=row["dataset_ties"],
                dataset_losses=row["dataset_losses"],
                inference_mode=row["inference_mode"],
                suite=suite,
                p_value=row["primary_p_value"],
                p_value_holm=row["primary_p_value_holm"],
            )
        )
    for row in controlled or []:
        output.append(
            _clean_result_row(
                result_kind="comparison",
                metric=row["candidate_metric"],
                reference_metric=row["reference_metric"],
                condition="controlled",
                anomaly_prevalence=row["target_anomaly_prevalence"],
                dataset_count=row["dataset_count"],
                split_count=row["paired_split_count"],
                selection=row["normalized_regret_reduction"],
                selection_low=row["normalized_regret_reduction_ci_low"],
                selection_high=row["normalized_regret_reduction_ci_high"],
                average_precision=row["average_precision_delta"],
                average_precision_low=row["average_precision_delta_ci_low"],
                average_precision_high=row["average_precision_delta_ci_high"],
                selection_efficiency=row["selection_efficiency_delta"],
                selection_efficiency_low=row["selection_efficiency_delta_ci_low"],
                selection_efficiency_high=row[
                    "selection_efficiency_delta_ci_high"
                ],
                near_oracle=row["near_oracle_rate_delta"],
                near_oracle_low=row["near_oracle_rate_delta_ci_low"],
                near_oracle_high=row["near_oracle_rate_delta_ci_high"],
                oracle_hit=row["oracle_hit_rate_delta"],
                oracle_hit_low=row["oracle_hit_rate_delta_ci_low"],
                oracle_hit_high=row["oracle_hit_rate_delta_ci_high"],
                success_rate=row["success_rate_delta"],
                success_rate_low=row["success_rate_delta_ci_low"],
                success_rate_high=row["success_rate_delta_ci_high"],
                runtime_seconds=row["runtime_reduction_seconds"],
                runtime_seconds_low=row["runtime_reduction_seconds_ci_low"],
                runtime_seconds_high=row["runtime_reduction_seconds_ci_high"],
                inference_mode="descriptive",
                suite=suite,
            )
        )
    return sorted(
        output,
        key=lambda row: (
            row["condition"] != "original",
            -float(row["selection_quality_effect"]),
            str(row["reference_metric"]),
        ),
    )


def _absolute_results_rows(
    original: list[dict[str, Any]],
    controlled: list[dict[str, Any]],
    *,
    suite: str,
    inference_mode: str,
) -> list[dict[str, Any]]:
    """Create the single concise result table for a standalone run."""
    output: list[dict[str, Any]] = []
    for row in original:
        output.append(
            _clean_result_row(
                result_kind="absolute",
                metric=row["metric"],
                reference_metric="",
                condition="original",
                anomaly_prevalence="source",
                dataset_count=row["dataset_count"],
                split_count=row["successful_splits"] + row["failed_splits"],
                selection=1.0 - row["mean_normalized_average_precision_regret"],
                selection_low=(
                    1.0
                    - row["mean_normalized_average_precision_regret_ci_high"]
                ),
                selection_high=(
                    1.0 - row["mean_normalized_average_precision_regret_ci_low"]
                ),
                average_precision=row["mean_selected_average_precision"],
                average_precision_low=row["mean_selected_average_precision_ci_low"],
                average_precision_high=row[
                    "mean_selected_average_precision_ci_high"
                ],
                selection_efficiency=row["mean_selection_efficiency_over_random"],
                selection_efficiency_low=row[
                    "mean_selection_efficiency_over_random_ci_low"
                ],
                selection_efficiency_high=row[
                    "mean_selection_efficiency_over_random_ci_high"
                ],
                roc_auc=row["mean_selected_roc_auc"],
                roc_auc_low=row["mean_selected_roc_auc_ci_low"],
                roc_auc_high=row["mean_selected_roc_auc_ci_high"],
                near_oracle=row["near_oracle_rate"],
                near_oracle_low=row["near_oracle_rate_ci_low"],
                near_oracle_high=row["near_oracle_rate_ci_high"],
                oracle_hit=row["oracle_hit_rate"],
                oracle_hit_low=row["oracle_hit_rate_ci_low"],
                oracle_hit_high=row["oracle_hit_rate_ci_high"],
                spearman=row["mean_spearman_average_precision"],
                spearman_low=row["mean_spearman_average_precision_ci_low"],
                spearman_high=row["mean_spearman_average_precision_ci_high"],
                kendall=row["mean_kendall_average_precision"],
                kendall_low=row["mean_kendall_average_precision_ci_low"],
                kendall_high=row["mean_kendall_average_precision_ci_high"],
                success_rate=row["success_rate"],
                success_rate_low=row["success_rate_ci_low"],
                success_rate_high=row["success_rate_ci_high"],
                runtime_seconds=row["mean_metric_evaluation_seconds"],
                runtime_seconds_low=row["mean_metric_evaluation_seconds_ci_low"],
                runtime_seconds_high=row[
                    "mean_metric_evaluation_seconds_ci_high"
                ],
                overall_rank=row["overall_rank"],
                selected_detector_distribution=row[
                    "selected_detector_distribution"
                ],
                suite=suite,
                inference_mode=inference_mode,
            )
        )
    for row in controlled:
        output.append(
            _clean_result_row(
                result_kind="absolute",
                metric=row["metric"],
                reference_metric="",
                condition="controlled",
                anomaly_prevalence=row["target_anomaly_prevalence"],
                dataset_count=row["dataset_count"],
                split_count=row["successful_splits"] + row["failed_splits"],
                selection=1.0 - row["mean_normalized_average_precision_regret"],
                selection_low=(
                    1.0
                    - row["mean_normalized_average_precision_regret_ci_high"]
                ),
                selection_high=(
                    1.0 - row["mean_normalized_average_precision_regret_ci_low"]
                ),
                average_precision=row["mean_selected_average_precision"],
                average_precision_low=row["mean_selected_average_precision_ci_low"],
                average_precision_high=row[
                    "mean_selected_average_precision_ci_high"
                ],
                selection_efficiency=row["mean_selection_efficiency_over_random"],
                selection_efficiency_low=row[
                    "mean_selection_efficiency_over_random_ci_low"
                ],
                selection_efficiency_high=row[
                    "mean_selection_efficiency_over_random_ci_high"
                ],
                near_oracle=row["near_oracle_rate"],
                near_oracle_low=row["near_oracle_rate_ci_low"],
                near_oracle_high=row["near_oracle_rate_ci_high"],
                oracle_hit=row["oracle_hit_rate"],
                oracle_hit_low=row["oracle_hit_rate_ci_low"],
                oracle_hit_high=row["oracle_hit_rate_ci_high"],
                spearman=row["mean_spearman_average_precision"],
                spearman_low=row["mean_spearman_average_precision_ci_low"],
                spearman_high=row["mean_spearman_average_precision_ci_high"],
                kendall=row["mean_kendall_average_precision"],
                kendall_low=row["mean_kendall_average_precision_ci_low"],
                kendall_high=row["mean_kendall_average_precision_ci_high"],
                success_rate=row["success_rate"],
                success_rate_low=row["success_rate_ci_low"],
                success_rate_high=row["success_rate_ci_high"],
                runtime_seconds=row["mean_metric_evaluation_seconds"],
                runtime_seconds_low=row["mean_metric_evaluation_seconds_ci_low"],
                runtime_seconds_high=row[
                    "mean_metric_evaluation_seconds_ci_high"
                ],
                overall_rank=row["overall_rank"],
                inference_mode="descriptive",
                suite=suite,
            )
        )
    return sorted(
        output,
        key=lambda row: (
            row["condition"] != "original",
            float(row["overall_rank"]),
            str(row["metric"]),
        ),
    )


def _clean_result_row(
    *,
    result_kind: str,
    metric: Any,
    reference_metric: Any,
    condition: str,
    anomaly_prevalence: Any,
    dataset_count: Any,
    split_count: Any,
    selection: Any,
    selection_low: Any,
    selection_high: Any,
    average_precision: Any,
    average_precision_low: Any,
    average_precision_high: Any,
    selection_efficiency: Any = "",
    selection_efficiency_low: Any = "",
    selection_efficiency_high: Any = "",
    roc_auc: Any = "",
    roc_auc_low: Any = "",
    roc_auc_high: Any = "",
    near_oracle: Any = "",
    near_oracle_low: Any = "",
    near_oracle_high: Any = "",
    oracle_hit: Any = "",
    oracle_hit_low: Any = "",
    oracle_hit_high: Any = "",
    spearman: Any = "",
    spearman_low: Any = "",
    spearman_high: Any = "",
    kendall: Any = "",
    kendall_low: Any = "",
    kendall_high: Any = "",
    success_rate: Any = "",
    success_rate_low: Any = "",
    success_rate_high: Any = "",
    runtime_seconds: Any = "",
    runtime_seconds_low: Any = "",
    runtime_seconds_high: Any = "",
    overall_rank: Any = "",
    selected_detector_distribution: Any = "",
    dataset_wins: Any = "",
    dataset_ties: Any = "",
    dataset_losses: Any = "",
    suite: Any = "",
    inference_mode: Any = "",
    p_value: Any = "",
    p_value_holm: Any = "",
) -> dict[str, Any]:
    row = {
        "result_kind": result_kind,
        "metric": metric,
        "reference_metric": reference_metric,
        "condition": condition,
        "anomaly_prevalence": anomaly_prevalence,
        "dataset_count": dataset_count,
        "split_count": split_count,
        "benchmark_version": __version__,
        "candidate_pool_version": CANDIDATE_POOL_VERSION,
        "suite": suite,
        "inference_mode": inference_mode,
        "overall_rank": overall_rank,
        "selection_quality_effect": selection,
        "selection_quality_ci_low": selection_low,
        "selection_quality_ci_high": selection_high,
        "average_precision_effect": average_precision,
        "average_precision_ci_low": average_precision_low,
        "average_precision_ci_high": average_precision_high,
        "selection_efficiency_effect": selection_efficiency,
        "selection_efficiency_ci_low": selection_efficiency_low,
        "selection_efficiency_ci_high": selection_efficiency_high,
        "roc_auc_effect": roc_auc,
        "roc_auc_ci_low": roc_auc_low,
        "roc_auc_ci_high": roc_auc_high,
        "near_oracle_effect": near_oracle,
        "near_oracle_ci_low": near_oracle_low,
        "near_oracle_ci_high": near_oracle_high,
        "oracle_hit_effect": oracle_hit,
        "oracle_hit_ci_low": oracle_hit_low,
        "oracle_hit_ci_high": oracle_hit_high,
        "spearman_effect": spearman,
        "spearman_ci_low": spearman_low,
        "spearman_ci_high": spearman_high,
        "kendall_effect": kendall,
        "kendall_ci_low": kendall_low,
        "kendall_ci_high": kendall_high,
        "success_rate_effect": success_rate,
        "success_rate_ci_low": success_rate_low,
        "success_rate_ci_high": success_rate_high,
        "runtime_effect_seconds": runtime_seconds,
        "runtime_ci_low": runtime_seconds_low,
        "runtime_ci_high": runtime_seconds_high,
        "selected_detector_distribution": selected_detector_distribution,
        "dataset_wins": dataset_wins,
        "dataset_ties": dataset_ties,
        "dataset_losses": dataset_losses,
        "primary_p_value": p_value,
        "primary_p_value_holm": p_value_holm,
    }
    return {key: _clean_display_value(value) for key, value in row.items()}


def _clean_display_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""
    return value


def _compare_prevalence_sensitivity(
    reference: Path,
    candidate: Path,
    candidate_metrics: list[str],
    reference_metrics: list[str],
) -> list[dict[str, Any]]:
    """Compare secondary controlled-prevalence effects without hypothesis tests."""
    reference_rows = _read_csv(reference / "prevalence_sensitivity.csv")
    candidate_rows = _read_csv(candidate / "prevalence_sensitivity.csv")
    targets = sorted(
        {
            float(row["target_anomaly_prevalence"])
            for row in reference_rows + candidate_rows
        }
    )
    effect_fields = (
        "normalized_regret_reduction",
        "selection_efficiency_delta",
        "average_precision_delta",
        "near_oracle_rate_delta",
        "oracle_hit_rate_delta",
        "success_rate_delta",
        "runtime_reduction_seconds",
    )
    output: list[dict[str, Any]] = []
    for candidate_metric in candidate_metrics:
        for reference_metric in reference_metrics:
            if candidate_metric == reference_metric:
                continue
            for target in targets:
                paired = _paired_prevalence_effects(
                    candidate_rows,
                    reference_rows,
                    candidate_metric,
                    reference_metric,
                    target,
                )
                if not paired:
                    continue
                result: dict[str, Any] = {
                    "candidate_metric": candidate_metric,
                    "reference_metric": reference_metric,
                    "target_anomaly_prevalence": target,
                    "dataset_count": len({row["dataset"] for row in paired}),
                    "paired_split_count": len(paired),
                }
                dataset_effects = np.asarray(
                    [
                        _mean(
                            [row for row in paired if row["dataset"] == dataset],
                            "normalized_regret_reduction",
                        )
                        for dataset in sorted({row["dataset"] for row in paired})
                    ],
                    dtype=float,
                )
                result["dataset_wins"] = int(np.sum(dataset_effects > 1e-15))
                result["dataset_ties"] = int(
                    np.sum(np.isclose(dataset_effects, 0.0, atol=1e-15))
                )
                result["dataset_losses"] = int(np.sum(dataset_effects < -1e-15))
                for effect in effect_fields:
                    mean, low, high = hierarchical_mean_ci(
                        paired,
                        effect,
                        rounds=BOOTSTRAP_ROUNDS,
                        seed=stable_seed(
                            candidate_metric,
                            reference_metric,
                            str(target),
                            effect,
                        ),
                    )
                    result[effect] = mean
                    result[f"{effect}_ci_low"] = low
                    result[f"{effect}_ci_high"] = high
                output.append(result)
    return output


def _paired_prevalence_effects(
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    candidate_metric: str,
    reference_metric: str,
    target: float,
) -> list[dict[str, Any]]:
    def index(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, str], Any]:
        return {
            (row["dataset"], row["seed"]): row
            for row in rows
            if row["metric"] == metric
            and float(row["target_anomaly_prevalence"]) == target
        }

    candidate = index(candidate_rows, candidate_metric)
    reference = index(reference_rows, reference_metric)
    output: list[dict[str, Any]] = []
    for dataset, seed in sorted(set(candidate) & set(reference)):
        new = candidate[(dataset, seed)]
        baseline = reference[(dataset, seed)]
        output.append(
            {
                "dataset": dataset,
                "seed": seed,
                "normalized_regret_reduction": (
                    _numeric(baseline["normalized_average_precision_regret"])
                    - _numeric(new["normalized_average_precision_regret"])
                ),
                "average_precision_delta": (
                    _numeric(new["selected_average_precision"])
                    - _numeric(baseline["selected_average_precision"])
                ),
                "selection_efficiency_delta": (
                    _numeric(new["selection_efficiency_over_random"])
                    - _numeric(baseline["selection_efficiency_over_random"])
                ),
                "near_oracle_rate_delta": (
                    _numeric(new["near_oracle"]) - _numeric(baseline["near_oracle"])
                ),
                "oracle_hit_rate_delta": (
                    _numeric(new["oracle_hit"]) - _numeric(baseline["oracle_hit"])
                ),
                "success_rate_delta": (
                    float(new["status"] == "ok")
                    - float(baseline["status"] == "ok")
                ),
                "runtime_reduction_seconds": (
                    _numeric(baseline["metric_evaluation_seconds"])
                    - _numeric(new["metric_evaluation_seconds"])
                ),
            }
        )
    return output


def _paired_effects(
    candidate_rows: list[dict[str, str]],
    reference_rows: list[dict[str, str]],
    candidate_metric: str,
    reference_metric: str,
) -> list[dict[str, Any]]:
    def index(rows: list[dict[str, str]], metric: str):
        return {
            (row["dataset"], row["seed"]): row
            for row in rows
            if row["metric"] == metric
        }

    candidate = index(candidate_rows, candidate_metric)
    reference = index(reference_rows, reference_metric)
    output: list[dict[str, Any]] = []
    for dataset, seed in sorted(set(candidate) & set(reference)):
        new = candidate[(dataset, seed)]
        baseline = reference[(dataset, seed)]
        output.append(
            {
                "dataset": dataset,
                "seed": seed,
                "normalized_regret_reduction": (
                    _numeric(baseline["normalized_average_precision_regret"])
                    - _numeric(new["normalized_average_precision_regret"])
                ),
                "average_precision_delta": (
                    _numeric(new["selected_average_precision"])
                    - _numeric(baseline["selected_average_precision"])
                ),
                "selection_efficiency_delta": (
                    _numeric(new["selection_efficiency_over_random"])
                    - _numeric(baseline["selection_efficiency_over_random"])
                ),
                "roc_auc_delta": (
                    _numeric(new["selected_roc_auc"])
                    - _numeric(baseline["selected_roc_auc"])
                ),
                "spearman_average_precision_delta": (
                    _numeric(new["spearman_average_precision"])
                    - _numeric(baseline["spearman_average_precision"])
                ),
                "kendall_average_precision_delta": (
                    _numeric(new["kendall_average_precision"])
                    - _numeric(baseline["kendall_average_precision"])
                ),
                "near_oracle_rate_delta": (
                    _numeric(new["near_oracle"]) - _numeric(baseline["near_oracle"])
                ),
                "oracle_hit_rate_delta": (
                    _numeric(new["oracle_hit"]) - _numeric(baseline["oracle_hit"])
                ),
                "success_rate_delta": (
                    float(new["status"] == "ok")
                    - float(baseline["status"] == "ok")
                ),
                "runtime_reduction_seconds": (
                    _numeric(baseline["metric_evaluation_seconds"])
                    - _numeric(new["metric_evaluation_seconds"])
                ),
            }
        )
    return output


def _comparison_dataset_rows(
    paired: list[dict[str, Any]],
    candidate_metric: str,
    reference_metric: str,
    effect_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in paired}):
        rows = [row for row in paired if row["dataset"] == dataset]
        result: dict[str, Any] = {
            "candidate_metric": candidate_metric,
            "reference_metric": reference_metric,
            "dataset": dataset,
            "paired_seed_count": len(rows),
        }
        for effect in effect_fields:
            result[effect] = _mean(rows, effect)
        output.append(result)
    return output


def run_benchmark(
    config: RunConfig,
    output_dir: str | Path,
    *,
    suite: str = "development",
    inference_mode: str = "exploratory",
) -> RunResult:
    """Run all configured datasets, seeds, candidates, and metrics."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    evidence_dir = artifacts / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(config.metrics)
    selected_datasets = tuple(
        dataset
        for dataset in config.datasets
        if suite == "all" or dataset.suite == suite
    )
    if not selected_datasets:
        raise ValueError(f"suite {suite!r} contains no configured datasets")
    loaded_datasets = []
    dataset_hashes: dict[str, str] = {}
    registry = {row["name"]: row for row in dataset_registry()}
    dataset_metadata: dict[str, dict[str, str]] = {}
    for dataset in selected_datasets:
        if (
            dataset.name in registry
            and dataset.suite != registry[dataset.name]["suite"]
        ):
            raise ValueError(
                f"dataset {dataset.name!r} has a noncanonical suite assignment"
            )
        X, y = load_dataset(
            dataset.name,
            path=dataset.path,
            data_dir=config.data_dir,
            verify_checksum=dataset.name != "synthetic",
        )
        identity = dataset_hash(X, y)
        dataset_hashes[dataset.name] = identity
        dataset_metadata[dataset.name] = dict(
            registry.get(
                dataset.name,
                {
                    "name": dataset.name,
                    "category": "synthetic",
                    "suite": dataset.suite,
                },
            )
        )
        loaded_datasets.append((dataset, X, y))
    config_hash = _config_hash(config, dataset_hashes, suite=suite)
    experiment_hash = _experiment_hash(config, dataset_hashes, suite=suite)
    candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    split_records: dict[str, dict[str, Any]] = {}

    for dataset, X, y in loaded_datasets:
        for seed in config.seeds:
            split = make_benchmark_split(
                X,
                y,
                seed=seed,
                test_fraction=config.test_fraction,
                max_normal_samples=config.max_normal_samples,
            )
            fit, validation = split_development_normals(
                split.development_normals,
                seed=seed,
                validation_fraction=config.validation_fraction,
            )
            specs = _candidate_specs(
                config,
                fit_size=len(fit),
                feature_count=fit.shape[1],
            )
            split_key = f"{dataset.name}:seed={seed}"
            evidence_path = evidence_dir / f"{dataset.name}__seed-{seed}.npz"
            split_cache_dir = artifacts / "splits" / f"{dataset.name}__seed-{seed}"
            cached = _load_split_cache(
                split_cache_dir,
                evidence_path,
                experiment_hash=experiment_hash,
                dataset_hash_value=dataset_hashes[dataset.name],
                split_hash=split.split_hash,
                candidate_count=len(specs),
            )
            if cached is None:
                rows, evidence, metric_seconds = _evaluate_candidates(
                    specs,
                    dataset=dataset.name,
                    seed=seed,
                    fit=fit,
                    validation=validation,
                    development=split.development_normals,
                    test_features=split.test_features,
                    test_labels=split.test_labels,
                    metrics=metrics,
                    detector_contamination=config.detector_contamination,
                    metric_tail_fraction=config.metric_tail_fraction,
                    reference_size=config.reference_size,
                    stability_repeats=config.stability_repeats,
                    controlled_prevalences=config.controlled_prevalences,
                )
                for row in rows:
                    row["dataset_hash"] = dataset_hashes[dataset.name]
                    row["split_hash"] = split.split_hash
                    row["candidate_pool_version"] = CANDIDATE_POOL_VERSION
                evidence_hash = _write_split_cache(
                    split_cache_dir,
                    evidence_path,
                    rows,
                    evidence,
                    experiment_hash=experiment_hash,
                    dataset_hash_value=dataset_hashes[dataset.name],
                    split_hash=split.split_hash,
                )
            else:
                rows, evidence, evidence_hash = cached
                metric_seconds = _apply_metrics(rows, evidence, metrics)
            for row in rows:
                row["dataset_hash"] = dataset_hashes[dataset.name]
                row["split_hash"] = split.split_hash
                row["candidate_pool_version"] = CANDIDATE_POOL_VERSION
                row["dataset_category"] = dataset_metadata[dataset.name]["category"]
            split_records[split_key] = {
                "dataset_hash": dataset_hashes[dataset.name],
                "split_hash": split.split_hash,
                "evidence_path": str(evidence_path.relative_to(output)),
                "evidence_hash": evidence_hash,
                "development_normal_count": len(split.development_normals),
                "test_count": len(split.test_labels),
                "test_anomaly_count": int(split.test_labels.sum()),
            }
            candidate_rows.extend(rows)
            selection_rows.extend(
                _select(
                    rows,
                    metrics,
                    dataset.name,
                    seed,
                    metric_seconds=metric_seconds,
                    include_reference_baselines=True,
                )
            )
            prevalence_rows.extend(
                _controlled_prevalence_rows(
                    rows,
                    metrics,
                    dataset=dataset.name,
                    seed=seed,
                    target_prevalences=config.controlled_prevalences,
                    metric_seconds=metric_seconds,
                    include_reference_baselines=True,
                )
            )
            budget_rows.extend(
                _budget_sensitivity_rows(
                    rows,
                    evidence,
                    metrics,
                    dataset.name,
                    seed,
                    include_reference_baselines=True,
                )
            )

    result = _write_result_tables(
        output,
        candidate_rows,
        selection_rows,
        prevalence_rows,
        budget_rows,
        suite=suite,
        inference_mode=inference_mode,
    )
    _write_manifest(
        artifacts / "manifest.json",
        config,
        config_hash,
        dataset_hashes=dataset_hashes,
        dataset_metadata=dataset_metadata,
        split_records=split_records,
        suite=suite,
        inference_mode=inference_mode,
    )
    return result


def _write_result_tables(
    output: Path,
    candidate_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    prevalence_rows: list[dict[str, Any]],
    budget_rows: list[dict[str, Any]],
    *,
    suite: str,
    inference_mode: str,
) -> RunResult:
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _write_csv(artifacts / "candidates.csv", candidate_rows)
    _write_csv(artifacts / "selections.csv", selection_rows)
    dataset_summary_rows = _dataset_summarize(selection_rows)
    _write_csv(artifacts / "dataset_summary.csv", dataset_summary_rows)
    _write_csv(artifacts / "domain_summary.csv", _domain_summarize(selection_rows))
    _write_csv(
        artifacts / "leave_one_domain_out.csv",
        _leave_one_domain_out(selection_rows),
    )
    summary_rows = _summarize(selection_rows)
    _write_csv(artifacts / "summary.csv", summary_rows)
    prevalence_sensitivity_rows = _prevalence_sensitivity(prevalence_rows)
    _write_csv(
        artifacts / "prevalence_sensitivity.csv", prevalence_sensitivity_rows
    )
    prevalence_summary_rows = _prevalence_summarize(prevalence_sensitivity_rows)
    _write_csv(
        artifacts / "prevalence_summary.csv",
        prevalence_summary_rows,
    )
    _write_csv(artifacts / "budget_sensitivity.csv", budget_rows)
    _write_csv(artifacts / "budget_summary.csv", _budget_summarize(budget_rows))
    _write_csv(
        output / "results.csv",
        _absolute_results_rows(
            summary_rows,
            prevalence_summary_rows,
            suite=suite,
            inference_mode=inference_mode,
        ),
    )
    return RunResult(
        output,
        output / "results.csv",
        artifacts / "candidates.csv",
        artifacts / "selections.csv",
        artifacts / "dataset_summary.csv",
        artifacts / "summary.csv",
        artifacts / "prevalence_sensitivity.csv",
        artifacts / "prevalence_summary.csv",
    )


def evaluate_metrics_from_reference(
    config: RunConfig,
    reference_dir: str | Path,
    output_dir: str | Path,
) -> RunResult:
    """Evaluate configured metrics from frozen normal-only reference evidence."""
    reference = Path(reference_dir)
    reference_artifacts = reference / "artifacts"
    reference_manifest_path = reference_artifacts / "manifest.json"
    reference_manifest = json.loads(reference_manifest_path.read_text())
    if reference_manifest.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("reference schema is incompatible; regenerate the reference")
    if reference_manifest.get("run_kind") != "reference":
        raise ValueError("metrics must be evaluated from a reference run")
    suite = str(reference_manifest["suite"])
    inference_mode = str(reference_manifest["inference_mode"])
    dataset_hashes = {
        str(name): str(value)
        for name, value in reference_manifest["dataset_hashes"].items()
    }
    dataset_metadata = reference_manifest["dataset_metadata"]
    expected_hash = _experiment_hash(config, dataset_hashes, suite=suite)
    if reference_manifest.get("experiment_hash") != expected_hash:
        raise ValueError(
            "reference and metric configuration use different benchmark settings"
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts = output / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(config.metrics)
    reference_candidates = _read_csv(reference_artifacts / "candidates.csv")
    candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    groups = sorted(
        {(row["dataset"], int(row["seed"])) for row in reference_candidates}
    )
    for dataset, seed in groups:
        split_key = f"{dataset}:seed={seed}"
        split_record = reference_manifest["splits"][split_key]
        evidence_path = reference / split_record["evidence_path"]
        if _sha256_file(evidence_path) != split_record["evidence_hash"]:
            raise ValueError(f"frozen evidence hash mismatch for {split_key}")
        evidence = _load_evidence(evidence_path)
        rows = [
            _candidate_base_row(row)
            for row in reference_candidates
            if row["dataset"] == dataset and int(row["seed"]) == seed
        ]
        rows.sort(key=lambda row: int(row["candidate_id"]))
        if len(rows) != len(evidence.anomaly_scores):
            raise ValueError(f"candidate/evidence size mismatch for {split_key}")
        metric_seconds = _apply_metrics(rows, evidence, metrics)
        candidate_rows.extend(rows)
        selection_rows.extend(
            _select(
                rows,
                metrics,
                dataset,
                seed,
                metric_seconds=metric_seconds,
            )
        )
        prevalence_rows.extend(
            _controlled_prevalence_rows(
                rows,
                metrics,
                dataset=dataset,
                seed=seed,
                target_prevalences=config.controlled_prevalences,
                metric_seconds=metric_seconds,
            )
        )
        budget_rows.extend(
            _budget_sensitivity_rows(rows, evidence, metrics, dataset, seed)
        )

    result = _write_result_tables(
        output,
        candidate_rows,
        selection_rows,
        prevalence_rows,
        budget_rows,
        suite=suite,
        inference_mode=inference_mode,
    )
    config_hash = _config_hash(config, dataset_hashes, suite=suite)
    _write_manifest(
        artifacts / "manifest.json",
        config,
        config_hash,
        dataset_hashes=dataset_hashes,
        dataset_metadata=dataset_metadata,
        split_records=reference_manifest["splits"],
        suite=suite,
        inference_mode=inference_mode,
        reference_manifest_hash=_sha256_file(reference_manifest_path),
        run_kind="metric_evaluation",
    )
    return result


def _candidate_base_row(row: dict[str, str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith(("metric_", "error_"))
    }


def _candidate_specs(
    config: RunConfig,
    *,
    fit_size: int,
    feature_count: int,
) -> list[CandidateSpec]:
    return candidate_specs(
        config.models,
        trials=config.trials,
        fit_size=fit_size,
        feature_count=feature_count,
    )


def _evaluate_candidates(
    specs: list[CandidateSpec],
    *,
    dataset: str,
    seed: int,
    fit: np.ndarray,
    validation: np.ndarray,
    development: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    metrics: tuple[MetricSpec, ...],
    detector_contamination: float,
    metric_tail_fraction: float,
    reference_size: int,
    stability_repeats: int,
    controlled_prevalences: tuple[float, ...],
) -> tuple[list[dict[str, Any]], EvidenceBundle, dict[str, float]]:
    validation_scaler = StandardScaler().fit(fit)
    fit_scaled = validation_scaler.transform(fit)
    validation_scaled = validation_scaler.transform(validation)
    low = validation_scaled.min(axis=0)
    high = validation_scaled.max(axis=0)
    reference = np.random.default_rng(seed + 1_000_000).uniform(
        low, high, size=(reference_size, validation_scaled.shape[1])
    )
    support_volume = float(bounding_box_volume(validation_scaled))

    final_scaler = StandardScaler().fit(development)
    development_scaled = final_scaler.transform(development)
    test_scaled = final_scaler.transform(test_features)

    rows: list[dict[str, Any]] = []
    anomaly_score_rows: list[np.ndarray] = []
    reference_score_rows: list[np.ndarray] = []
    repeated_score_rows: list[np.ndarray] = []
    probability_rows: list[np.ndarray] = []
    for candidate_index, spec in enumerate(specs):
        candidate_seed = seed * 10_000 + candidate_index
        started = time.perf_counter()
        detector = build_detector(
            spec,
            seed=candidate_seed,
            detector_contamination=detector_contamination,
        )
        fit_detector(detector, fit_scaled)
        validation_scores = _finite_scores(
            detector.decision_function(validation_scaled), spec
        )
        reference_scores = _finite_scores(detector.decision_function(reference), spec)
        outlier_probabilities = detector.predict_proba(validation_scaled)[:, 1]
        repeated_scores = [validation_scores]
        for repeat in range(1, stability_repeats):
            bootstrap_rng = np.random.default_rng(candidate_seed + repeat)
            bootstrap_indices = bootstrap_rng.choice(
                len(fit_scaled), len(fit_scaled), replace=True
            )
            repeated_detector = build_detector(
                spec,
                seed=candidate_seed + repeat,
                detector_contamination=detector_contamination,
            )
            fit_detector(repeated_detector, fit_scaled[bootstrap_indices])
            repeated_scores.append(
                _finite_scores(
                    repeated_detector.decision_function(validation_scaled), spec
                )
            )
        fit_seconds = time.perf_counter() - started

        final_detector = build_detector(
            spec,
            seed=candidate_seed,
            detector_contamination=detector_contamination,
        )
        fit_detector(final_detector, development_scaled)
        test_scores = _finite_scores(
            final_detector.decision_function(test_scaled), spec
        )
        (
            test_average_precision,
            test_roc_auc,
            test_anomaly_prevalence,
        ) = _supervised_scores(test_labels, test_scores)
        row: dict[str, Any] = {
            "dataset": dataset,
            "seed": seed,
            "candidate_id": candidate_index,
            "detector": spec.detector,
            "is_default": spec.is_default,
            "params": json.dumps(spec.params, sort_keys=True),
            "validation_fit_seconds": fit_seconds,
            "test_roc_auc": test_roc_auc,
            "test_average_precision": test_average_precision,
            "test_anomaly_prevalence": test_anomaly_prevalence,
            "test_normal_count": int(np.sum(test_labels == 0)),
            "test_anomaly_count": int(np.sum(test_labels == 1)),
            "controlled_average_precision": json.dumps(
                {
                    str(target): _prior_adjusted_average_precision(
                        test_labels, test_scores, target
                    )
                    for target in controlled_prevalences
                },
                sort_keys=True,
            ),
        }
        rows.append(row)
        anomaly_score_rows.append(np.asarray(validation_scores, dtype=float))
        reference_score_rows.append(np.asarray(reference_scores, dtype=float))
        repeated_score_rows.append(np.asarray(repeated_scores, dtype=float))
        probability_rows.append(np.asarray(outlier_probabilities, dtype=float))

    evidence = EvidenceBundle(
        validation_features=np.asarray(validation_scaled, dtype=float),
        anomaly_scores=np.asarray(anomaly_score_rows, dtype=float),
        reference_scores=np.asarray(reference_score_rows, dtype=float),
        repeated_anomaly_scores=np.asarray(repeated_score_rows, dtype=float),
        outlier_probabilities=np.asarray(probability_rows, dtype=float),
        support_volume=support_volume,
        metric_tail_fraction=metric_tail_fraction,
    )
    metric_seconds = _apply_metrics(rows, evidence, metrics)
    return rows, evidence, metric_seconds


def _finite_scores(scores: np.ndarray, spec: CandidateSpec) -> np.ndarray:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError(
            f"{spec.detector} candidate produced non-finite decision scores: "
            f"{json.dumps(spec.params, sort_keys=True)}"
        )
    return values


def _apply_metrics(
    rows: list[dict[str, Any]],
    evidence: EvidenceBundle,
    metrics: tuple[MetricSpec, ...],
) -> dict[str, float]:
    metric_inputs = evidence.metric_inputs()
    runtimes: dict[str, float] = {}
    for metric in metrics:
        started = time.perf_counter()
        try:
            values = metric.evaluate(metric_inputs)
            for row, value in zip(rows, values, strict=True):
                row[f"metric_{metric.name}"] = float(value)
                row[f"error_{metric.name}"] = ""
        except Exception as exc:  # Failures are retained and penalized.
            error = f"{type(exc).__name__}: {exc}"
            for row in rows:
                row[f"metric_{metric.name}"] = ""
                row[f"error_{metric.name}"] = error
        runtimes[metric.name] = time.perf_counter() - started
        for row in rows:
            row[f"metric_seconds_{metric.name}"] = runtimes[metric.name]
    return runtimes


def _prior_adjusted_average_precision(
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
    target_prevalence: float,
) -> float:
    """Compute empirical AP under prior shift without stochastic subsampling."""
    labels = np.asarray(labels, dtype=int)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("prior-adjusted AP requires both classes")
    weights = np.where(
        labels == 1,
        target_prevalence / positives,
        (1.0 - target_prevalence) / negatives,
    )
    return _unit_interval(
        average_precision_score(labels, anomaly_scores, sample_weight=weights)
    )


def _select(
    rows: list[dict[str, Any]],
    metrics: tuple[MetricSpec, ...],
    dataset: str,
    seed: int,
    *,
    metric_seconds: dict[str, float] | None = None,
    include_reference_baselines: bool = False,
) -> list[dict[str, Any]]:
    average_precision = np.asarray(
        [float(row["test_average_precision"]) for row in rows]
    )
    oracle_average_precision = float(np.max(average_precision))
    worst_average_precision = float(np.min(average_precision))
    mean_average_precision = float(np.mean(average_precision))
    performance_range = oracle_average_precision - worst_average_precision
    supervised_ranks = rankdata(average_precision, method="average")
    if len(rows) == 1 or np.isclose(performance_range, 0.0):
        supervised_percentiles = np.ones(len(rows), dtype=float)
    else:
        supervised_percentiles = (supervised_ranks - 1.0) / (len(rows) - 1.0)

    output: list[dict[str, Any]] = []
    for metric in metrics:
        key = f"metric_{metric.name}"
        valid = [row for row in rows if row[key] != ""]
        if not valid:
            random_headroom = oracle_average_precision - mean_average_precision
            output.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "metric": metric.name,
                    "status": "failed",
                    "success": False,
                    "failure_reason": rows[0].get(f"error_{metric.name}", ""),
                    "selected_candidate_id": "",
                    "selected_detector": "",
                    "selected_average_precision": worst_average_precision,
                    "oracle_average_precision": oracle_average_precision,
                    "worst_average_precision": worst_average_precision,
                    "average_precision_regret": (
                        oracle_average_precision - worst_average_precision
                    ),
                    "normalized_average_precision_regret": 1.0,
                    "average_precision_lift_over_mean": (
                        worst_average_precision - mean_average_precision
                    ),
                    "selection_efficiency_over_random": (
                        0.0
                        if np.isclose(random_headroom, 0.0)
                        else (worst_average_precision - mean_average_precision)
                        / random_headroom
                    ),
                    "selected_supervised_percentile": 0.0,
                    "near_oracle": False,
                    "oracle_hit": False,
                    "selected_roc_auc": min(
                        float(row["test_roc_auc"]) for row in rows
                    ),
                    "spearman_average_precision": -1.0,
                    "kendall_average_precision": -1.0,
                    "metric_evaluation_seconds": (metric_seconds or {}).get(
                        metric.name, 0.0
                    ),
                    "candidate_pool_version": CANDIDATE_POOL_VERSION,
                }
            )
            continue
        reverse = metric.direction == "maximize"
        selected = sorted(valid, key=lambda row: float(row[key]), reverse=reverse)[0]
        selected_index = rows.index(selected)
        selected_average_precision = float(selected["test_average_precision"])
        average_precision_regret = max(
            0.0, oracle_average_precision - selected_average_precision
        )
        normalized_regret = (
            0.0
            if np.isclose(performance_range, 0.0)
            else average_precision_regret / performance_range
        )
        signal = np.array([float(row[key]) for row in valid])
        quality = np.array([float(row["test_average_precision"]) for row in valid])
        if metric.direction == "minimize":
            signal = -signal
        has_rank_signal = np.unique(signal).size > 1 and np.unique(quality).size > 1
        spearman = spearmanr(signal, quality).statistic if has_rank_signal else np.nan
        kendall = kendalltau(signal, quality).statistic if has_rank_signal else np.nan
        output.append(
            {
                "dataset": dataset,
                "seed": seed,
                "metric": metric.name,
                "status": "ok",
                "success": True,
                "selected_candidate_id": selected["candidate_id"],
                "selected_detector": selected["detector"],
                "selected_average_precision": selected_average_precision,
                "oracle_average_precision": oracle_average_precision,
                "worst_average_precision": worst_average_precision,
                "average_precision_regret": average_precision_regret,
                "normalized_average_precision_regret": normalized_regret,
                "average_precision_lift_over_mean": (
                    selected_average_precision - mean_average_precision
                ),
                "selection_efficiency_over_random": (
                    0.0
                    if np.isclose(
                        oracle_average_precision - mean_average_precision, 0.0
                    )
                    else (selected_average_precision - mean_average_precision)
                    / (oracle_average_precision - mean_average_precision)
                ),
                "selected_supervised_percentile": float(
                    supervised_percentiles[selected_index]
                ),
                "near_oracle": bool(normalized_regret <= 0.05 + 1e-15),
                "oracle_hit": bool(
                    np.isclose(
                        selected_average_precision,
                        oracle_average_precision,
                        atol=1e-15,
                    )
                ),
                "selected_roc_auc": selected["test_roc_auc"],
                "spearman_average_precision": spearman,
                "kendall_average_precision": kendall,
                "failure_reason": "",
                "metric_evaluation_seconds": (metric_seconds or {}).get(
                    metric.name, 0.0
                ),
                "candidate_pool_version": CANDIDATE_POOL_VERSION,
            }
        )
    if include_reference_baselines:
        output.append(_random_selection_baseline(rows, dataset, seed))
        output.extend(_fixed_default_baselines(rows, dataset, seed))
    for result in output:
        result["dataset_category"] = rows[0].get("dataset_category", "synthetic")
    return output


def _random_selection_baseline(
    rows: list[dict[str, Any]], dataset: str, seed: int
) -> dict[str, Any]:
    average_precision = np.asarray(
        [float(row["test_average_precision"]) for row in rows], dtype=float
    )
    roc_auc = np.asarray([float(row["test_roc_auc"]) for row in rows], dtype=float)
    oracle = float(np.max(average_precision))
    worst = float(np.min(average_precision))
    mean = float(np.mean(average_precision))
    performance_range = oracle - worst
    regrets = oracle - average_precision
    normalized = (
        np.zeros_like(regrets)
        if np.isclose(performance_range, 0.0)
        else regrets / performance_range
    )
    return {
        "dataset": dataset,
        "seed": seed,
        "metric": "random_candidate",
        "status": "ok",
        "success": True,
        "selected_candidate_id": "expected",
        "selected_detector": "random",
        "selected_average_precision": mean,
        "oracle_average_precision": oracle,
        "worst_average_precision": worst,
        "average_precision_regret": oracle - mean,
        "normalized_average_precision_regret": float(np.mean(normalized)),
        "average_precision_lift_over_mean": 0.0,
        "selection_efficiency_over_random": 0.0,
        "selected_supervised_percentile": 0.5,
        "near_oracle": float(np.mean(normalized <= 0.05 + 1e-15)),
        "oracle_hit": float(np.mean(np.isclose(average_precision, oracle))),
        "selected_roc_auc": float(np.mean(roc_auc)),
        "spearman_average_precision": 0.0,
        "kendall_average_precision": 0.0,
        "failure_reason": "",
        "metric_evaluation_seconds": 0.0,
        "candidate_pool_version": CANDIDATE_POOL_VERSION,
    }


def _fixed_default_baselines(
    rows: list[dict[str, Any]], dataset: str, seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for selected_index, selected in enumerate(rows):
        if not bool(_numeric(selected.get("is_default", False))):
            continue
        name = f"default_{selected['detector']}"
        temporary = [dict(row) for row in rows]
        for index, row in enumerate(temporary):
            row[f"metric_{name}"] = float(index == selected_index)
            row[f"error_{name}"] = ""
        metric = MetricSpec(name, "maximize", compute=lambda data: 0.0)
        baseline = _select(temporary, (metric,), dataset, seed)[0]
        baseline["spearman_average_precision"] = 0.0
        baseline["kendall_average_precision"] = 0.0
        baseline["metric_evaluation_seconds"] = 0.0
        output.append(baseline)
    return output


CONTROLLED_SUMMARY_FIELDS = {
    "selected_average_precision": "mean_selected_average_precision",
    "oracle_average_precision": "mean_oracle_average_precision",
    "average_precision_regret": "mean_average_precision_regret",
    "normalized_average_precision_regret": (
        "mean_normalized_average_precision_regret"
    ),
    "average_precision_lift_over_mean": "mean_average_precision_lift_over_mean",
    "selection_efficiency_over_random": "mean_selection_efficiency_over_random",
    "selected_supervised_percentile": "mean_selected_supervised_percentile",
    "near_oracle": "near_oracle_rate",
    "oracle_hit": "oracle_hit_rate",
    "spearman_average_precision": "mean_spearman_average_precision",
    "kendall_average_precision": "mean_kendall_average_precision",
    "metric_evaluation_seconds": "mean_metric_evaluation_seconds",
    "success": "success_rate",
}


def _budget_sensitivity_rows(
    rows: list[dict[str, Any]],
    evidence: EvidenceBundle,
    metrics: tuple[MetricSpec, ...],
    dataset: str,
    seed: int,
    *,
    include_reference_baselines: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate nested balanced candidate budgets from defaults to the full pool."""
    model_count = len({str(row["detector"]) for row in rows})
    half = max(model_count, (len(rows) // (2 * model_count)) * model_count)
    budgets = sorted({model_count, half, len(rows)})
    output: list[dict[str, Any]] = []
    for budget in budgets:
        subset = [dict(row) for row in rows[:budget]]
        if budget < len(rows):
            inputs = _slice_evidence(evidence, budget).metric_inputs()
            for metric in metrics:
                if metric.pool_compute is None or metric.name == "ireos":
                    continue
                try:
                    values = metric.evaluate(inputs)
                    for row, value in zip(subset, values, strict=True):
                        row[f"metric_{metric.name}"] = float(value)
                        row[f"error_{metric.name}"] = ""
                except Exception as exc:
                    for row in subset:
                        row[f"metric_{metric.name}"] = ""
                        row[f"error_{metric.name}"] = (
                            f"{type(exc).__name__}: {exc}"
                        )
        selections = _select(
            subset,
            metrics,
            dataset,
            seed,
            include_reference_baselines=include_reference_baselines,
        )
        for selection in selections:
            selection["candidate_budget"] = budget
        output.extend(selections)
    return output


def _slice_evidence(evidence: EvidenceBundle, count: int) -> EvidenceBundle:
    return EvidenceBundle(
        validation_features=evidence.validation_features,
        anomaly_scores=evidence.anomaly_scores[:count],
        reference_scores=evidence.reference_scores[:count],
        repeated_anomaly_scores=evidence.repeated_anomaly_scores[:count],
        outlier_probabilities=evidence.outlier_probabilities[:count],
        support_volume=evidence.support_volume,
        metric_tail_fraction=evidence.metric_tail_fraction,
    )


def _budget_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {(str(row["metric"]), int(row["candidate_budget"])) for row in rows}
    )
    for metric, budget in groups:
        grouped = [
            row
            for row in rows
            if row["metric"] == metric and int(row["candidate_budget"]) == budget
        ]
        result: dict[str, Any] = {
            "metric": metric,
            "candidate_budget": budget,
            "dataset_count": len({row["dataset"] for row in grouped}),
            "split_count": len(grouped),
        }
        for field in (
            "normalized_average_precision_regret",
            "selection_efficiency_over_random",
            "selected_supervised_percentile",
            "success",
        ):
            mean, low, high = hierarchical_mean_ci(
                grouped,
                field,
                rounds=BOOTSTRAP_ROUNDS,
                seed=stable_seed(metric, str(budget), field),
            )
            result[f"mean_{field}"] = mean
            result[f"{field}_ci_low"] = low
            result[f"{field}_ci_high"] = high
        output.append(result)
    return output


def _controlled_prevalence_rows(
    rows: list[dict[str, Any]],
    metrics: tuple[MetricSpec, ...],
    *,
    dataset: str,
    seed: int,
    target_prevalences: tuple[float, ...],
    metric_seconds: dict[str, float] | None = None,
    include_reference_baselines: bool = False,
) -> list[dict[str, Any]]:
    """Evaluate selections under empirical class-prior adjustment."""
    output: list[dict[str, Any]] = []
    for target in target_prevalences:
        adjusted_rows = []
        for row in rows:
            adjusted = dict(row)
            values = json.loads(str(row["controlled_average_precision"]))
            adjusted["test_average_precision"] = float(values[str(target)])
            adjusted_rows.append(adjusted)
        selected = _select(
            adjusted_rows,
            metrics,
            dataset,
            seed,
            metric_seconds=metric_seconds,
            include_reference_baselines=include_reference_baselines,
        )
        for selection in selected:
            selection["target_anomaly_prevalence"] = target
            selection["prevalence_method"] = "prior_reweighting"
            selection["test_normal_count"] = rows[0]["test_normal_count"]
            selection["test_anomaly_count"] = rows[0]["test_anomaly_count"]
        output.extend(selected)
    return output


def _prevalence_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one prior-adjusted result per dataset, seed, metric, and target."""
    return rows


def _prevalence_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(row["metric"]), float(row["target_anomaly_prevalence"]))
            for row in rows
        }
    )
    for metric, target in groups:
        grouped = [
            row
            for row in rows
            if row["metric"] == metric
            and float(row["target_anomaly_prevalence"]) == target
        ]
        successful = [row for row in grouped if row["status"] == "ok"]
        result: dict[str, Any] = {
            "metric": metric,
            "target_anomaly_prevalence": target,
            "successful_splits": len(successful),
            "failed_splits": len(grouped) - len(successful),
            "failure_rate": (len(grouped) - len(successful)) / len(grouped),
            "dataset_count": len({row["dataset"] for row in grouped}),
        }
        for source, destination in CONTROLLED_SUMMARY_FIELDS.items():
            mean, low, high = hierarchical_mean_ci(
                grouped,
                source,
                rounds=BOOTSTRAP_ROUNDS,
                seed=stable_seed(metric, str(target), source),
            )
            result[destination] = mean
            result[f"{destination}_ci_low"] = low
            result[f"{destination}_ci_high"] = high
        output.append(result)
    for target in sorted({row["target_anomaly_prevalence"] for row in output}):
        target_rows = [
            row for row in output if row["target_anomaly_prevalence"] == target
        ]
        ranks = rankdata(
            [row["mean_normalized_average_precision_regret"] for row in target_rows],
            method="average",
        )
        for row, rank in zip(target_rows, ranks, strict=True):
            row["overall_rank"] = float(rank)
            row["selection_quality"] = (
                1.0 - row["mean_normalized_average_precision_regret"]
            )
    return output


SUMMARY_FIELDS = {
    "selected_average_precision": "mean_selected_average_precision",
    "selected_roc_auc": "mean_selected_roc_auc",
    "average_precision_regret": "mean_average_precision_regret",
    "normalized_average_precision_regret": ("mean_normalized_average_precision_regret"),
    "average_precision_lift_over_mean": "mean_average_precision_lift_over_mean",
    "selection_efficiency_over_random": "mean_selection_efficiency_over_random",
    "selected_supervised_percentile": "mean_selected_supervised_percentile",
    "near_oracle": "near_oracle_rate",
    "oracle_hit": "oracle_hit_rate",
    "spearman_average_precision": "mean_spearman_average_precision",
    "kendall_average_precision": "mean_kendall_average_precision",
    "metric_evaluation_seconds": "mean_metric_evaluation_seconds",
    "success": "success_rate",
}


def _dataset_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted({(str(row["metric"]), str(row["dataset"])) for row in rows})
    for metric, dataset in groups:
        grouped = [
            row for row in rows if row["metric"] == metric and row["dataset"] == dataset
        ]
        successful = [row for row in grouped if row["status"] == "ok"]
        result: dict[str, Any] = {
            "metric": metric,
            "dataset": dataset,
            "successful_seeds": len(successful),
            "failed_seeds": len(grouped) - len(successful),
            "failure_rate": (len(grouped) - len(successful)) / len(grouped),
        }
        for source, destination in SUMMARY_FIELDS.items():
            result[destination] = _mean(grouped, source)
        output.append(result)
    return output


def _domain_summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (str(row["metric"]), str(row["dataset_category"]))
            for row in rows
        }
    )
    for metric, category in groups:
        grouped = [
            row
            for row in rows
            if row["metric"] == metric and row["dataset_category"] == category
        ]
        output.append(
            {
                "metric": metric,
                "dataset_category": category,
                "dataset_count": len({row["dataset"] for row in grouped}),
                "split_count": len(grouped),
                "mean_normalized_average_precision_regret": _mean(
                    grouped, "normalized_average_precision_regret"
                ),
                "mean_selected_supervised_percentile": _mean(
                    grouped, "selected_supervised_percentile"
                ),
                "success_rate": _mean(grouped, "success"),
            }
        )
    return output


def _leave_one_domain_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = sorted({str(row["metric"]) for row in rows})
    categories = sorted({str(row["dataset_category"]) for row in rows})
    for metric in metrics:
        for held_out in categories:
            grouped = [
                row
                for row in rows
                if row["metric"] == metric
                and row["dataset_category"] != held_out
            ]
            mean, low, high = hierarchical_mean_ci(
                grouped,
                "normalized_average_precision_regret",
                rounds=BOOTSTRAP_ROUNDS,
                seed=stable_seed(metric, held_out, "leave-one-domain-out"),
            )
            output.append(
                {
                    "metric": metric,
                    "held_out_domain": held_out,
                    "dataset_count": len({row["dataset"] for row in grouped}),
                    "mean_normalized_average_precision_regret": mean,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return output


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_names = sorted({str(row["metric"]) for row in rows})
    for metric in metric_names:
        metric_rows = [row for row in rows if row["metric"] == metric]
        successful = [row for row in metric_rows if row["status"] == "ok"]
        result: dict[str, Any] = {
            "metric": metric,
            "successful_splits": len(successful),
            "failed_splits": len(metric_rows) - len(successful),
            "failure_rate": (len(metric_rows) - len(successful)) / len(metric_rows),
            "dataset_count": len({row["dataset"] for row in metric_rows}),
            "selected_detector_distribution": json.dumps(
                {
                    detector: sum(
                        row.get("selected_detector") == detector for row in successful
                    )
                    for detector in sorted(
                        {
                            str(row["selected_detector"])
                            for row in successful
                            if row.get("selected_detector")
                        }
                    )
                },
                sort_keys=True,
            ),
        }
        for source, destination in SUMMARY_FIELDS.items():
            mean, low, high = hierarchical_mean_ci(
                metric_rows,
                source,
                rounds=BOOTSTRAP_ROUNDS,
                seed=stable_seed(metric, source),
            )
            result[destination] = mean
            result[f"{destination}_ci_low"] = low
            result[f"{destination}_ci_high"] = high
        output.append(result)
    ranks = rankdata(
        [row["mean_normalized_average_precision_regret"] for row in output],
        method="average",
    )
    for row, rank in zip(output, ranks, strict=True):
        row["overall_rank"] = float(rank)
        row["selection_quality"] = (
            1.0 - row["mean_normalized_average_precision_regret"]
        )
    return output


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = np.array([_numeric(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def _numeric(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        if value.lower() == "true":
            return 1.0
        if value.lower() == "false":
            return 0.0
    return float(value)


def _unit_interval(value: float) -> float:
    bounded = float(np.clip(value, 0.0, 1.0))
    if np.isclose(bounded, 0.0, atol=1e-15):
        return 0.0
    if np.isclose(bounded, 1.0, atol=1e-15):
        return 1.0
    return bounded


def _supervised_scores(
    labels: np.ndarray, anomaly_scores: np.ndarray
) -> tuple[float, float, float]:
    return (
        _unit_interval(average_precision_score(labels, anomaly_scores)),
        _unit_interval(roc_auc_score(labels, anomaly_scores)),
        float(np.mean(labels)),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows produced for {path.name}")
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _config_hash(
    config: RunConfig,
    dataset_hashes: dict[str, str],
    *,
    suite: str,
) -> str:
    values = experiment_dict(config)
    values["metrics"] = list(config.metrics)
    values["suite"] = suite
    values["dataset_hashes"] = dataset_hashes
    values["candidate_pool_version"] = CANDIDATE_POOL_VERSION
    payload = json.dumps(values, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _experiment_hash(
    config: RunConfig,
    dataset_hashes: dict[str, str],
    *,
    suite: str,
) -> str:
    values = experiment_dict(config)
    values["suite"] = suite
    values["dataset_hashes"] = dataset_hashes
    values["candidate_pool_version"] = CANDIDATE_POOL_VERSION
    payload = json.dumps(values, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(
    path: Path,
    config: RunConfig,
    config_hash: str,
    *,
    dataset_hashes: dict[str, str],
    dataset_metadata: dict[str, dict[str, str]],
    split_records: dict[str, dict[str, Any]],
    suite: str,
    inference_mode: str,
    reference_manifest_hash: str | None = None,
    run_kind: str = "reference",
) -> None:
    payload = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "benchmark_version": __version__,
        "candidate_pool_version": CANDIDATE_POOL_VERSION,
        "run_kind": run_kind,
        "suite": suite,
        "inference_mode": inference_mode,
        "created_at": datetime.now(UTC).isoformat(),
        "label_free_bench_version": __version__,
        "git": _git_provenance(),
        "environment": _environment_provenance(),
        "config_hash": config_hash,
        "experiment_hash": _experiment_hash(
            config, dataset_hashes, suite=suite
        ),
        "dataset_hashes": dataset_hashes,
        "dataset_metadata": dataset_metadata,
        "splits": split_records,
        "reference_manifest_hash": reference_manifest_hash,
        "config": config_dict(config),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_evidence(path: Path, evidence: EvidenceBundle) -> str:
    np.savez_compressed(
        path,
        validation_features=evidence.validation_features,
        anomaly_scores=evidence.anomaly_scores,
        reference_scores=evidence.reference_scores,
        repeated_anomaly_scores=evidence.repeated_anomaly_scores,
        outlier_probabilities=evidence.outlier_probabilities,
        support_volume=np.asarray(evidence.support_volume),
        metric_tail_fraction=np.asarray(evidence.metric_tail_fraction),
    )
    return _sha256_file(path)


def _write_split_cache(
    split_dir: Path,
    evidence_path: Path,
    rows: list[dict[str, Any]],
    evidence: EvidenceBundle,
    *,
    experiment_hash: str,
    dataset_hash_value: str,
    split_hash: str,
) -> str:
    split_dir.mkdir(parents=True, exist_ok=True)
    evidence_hash = _write_evidence(evidence_path, evidence)
    candidate_path = split_dir / "candidates.csv"
    _write_csv(candidate_path, [_candidate_base_row(row) for row in rows])
    metadata = {
        "experiment_hash": experiment_hash,
        "dataset_hash": dataset_hash_value,
        "split_hash": split_hash,
        "candidate_pool_version": CANDIDATE_POOL_VERSION,
        "candidate_count": len(rows),
        "candidate_hash": _sha256_file(candidate_path),
        "evidence_hash": evidence_hash,
        "complete": True,
    }
    (split_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return evidence_hash


def _load_split_cache(
    split_dir: Path,
    evidence_path: Path,
    *,
    experiment_hash: str,
    dataset_hash_value: str,
    split_hash: str,
    candidate_count: int,
) -> tuple[list[dict[str, Any]], EvidenceBundle, str] | None:
    metadata_path = split_dir / "metadata.json"
    candidate_path = split_dir / "candidates.csv"
    if (
        not metadata_path.exists()
        or not candidate_path.exists()
        or not evidence_path.exists()
    ):
        return None
    metadata = json.loads(metadata_path.read_text())
    expected = {
        "experiment_hash": experiment_hash,
        "dataset_hash": dataset_hash_value,
        "split_hash": split_hash,
        "candidate_pool_version": CANDIDATE_POOL_VERSION,
        "candidate_count": candidate_count,
        "complete": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    if _sha256_file(candidate_path) != metadata.get("candidate_hash"):
        return None
    evidence_hash = _sha256_file(evidence_path)
    if evidence_hash != metadata.get("evidence_hash"):
        return None
    rows = [dict(row) for row in _read_csv(candidate_path)]
    if len(rows) != candidate_count:
        return None
    return rows, _load_evidence(evidence_path), evidence_hash


def _load_evidence(path: Path) -> EvidenceBundle:
    with np.load(path, allow_pickle=False) as archive:
        return EvidenceBundle(
            validation_features=np.asarray(
                archive["validation_features"], dtype=float
            ),
            anomaly_scores=np.asarray(archive["anomaly_scores"], dtype=float),
            reference_scores=np.asarray(archive["reference_scores"], dtype=float),
            repeated_anomaly_scores=np.asarray(
                archive["repeated_anomaly_scores"], dtype=float
            ),
            outlier_probabilities=np.asarray(
                archive["outlier_probabilities"], dtype=float
            ),
            support_volume=float(archive["support_volume"]),
            metric_tail_fraction=float(archive["metric_tail_fraction"]),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "revision": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _environment_provenance() -> dict[str, Any]:
    packages = (
        "labelfree",
        "numpy",
        "pyod",
        "scikit-learn",
        "scipy",
    )
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in packages
        },
    }
