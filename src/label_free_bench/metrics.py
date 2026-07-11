"""Metric protocol and adapters for every scalar ``labelfree`` metric."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Literal

import numpy as np
from labelfree.metrics import (
    asi_score,
    asoi_score,
    average_rank_consensus_scores,
    excess_mass_auc,
    expected_anomaly_gap_score,
    hits_model_scores,
    ireos_scores,
    laplacian_score,
    mass_volume_auc,
    model_centrality_scores,
    normalized_pseudo_discrepancy_score,
    ranking_stability_score,
    relative_top_median_score,
    score_cluster_metrics,
    sireos_score,
    top_k_stability_score,
)
from numpy.typing import NDArray

Direction = Literal["maximize", "minimize"]
CandidateCompute = Callable[["MetricInput"], float]
PoolCompute = Callable[[Sequence["MetricInput"]], NDArray[np.float64]]


@dataclass(frozen=True)
class MetricInput:
    """Normal-only evidence exposed to metric implementations."""

    validation_features: NDArray[np.float64]
    anomaly_scores: NDArray[np.float64]
    metric_tail_fraction: float
    reference_scores: NDArray[np.float64]
    support_volume: float
    repeated_anomaly_scores: NDArray[np.float64]
    outlier_probabilities: NDArray[np.float64]


@dataclass(frozen=True)
class MetricSpec:
    """A per-candidate or candidate-pool metric with a selection direction."""

    name: str
    direction: Direction
    compute: CandidateCompute | None = None
    pool_compute: PoolCompute | None = None

    def __post_init__(self) -> None:
        if (self.compute is None) == (self.pool_compute is None):
            raise ValueError("a metric needs exactly one compute function")

    def evaluate(self, inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
        if self.compute is not None:
            values = [self.compute(item) for item in inputs]
        else:
            assert self.pool_compute is not None
            values = self.pool_compute(inputs)
        result = np.asarray(values, dtype=float)
        if result.shape != (len(inputs),):
            raise ValueError(
                f"metric returned shape {result.shape}, expected {(len(inputs),)}"
            )
        if not np.isfinite(result).all():
            raise ValueError("metric returned a non-finite value")
        return result


def _clusters(data: MetricInput) -> dict[str, float]:
    return score_cluster_metrics(
        data.anomaly_scores, contamination=data.metric_tail_fraction
    )


def _silhouette(data: MetricInput) -> float:
    return float(_clusters(data)["silhouette"])


def _calinski_harabasz(data: MetricInput) -> float:
    return float(_clusters(data)["calinski_harabasz"])


def _davies_bouldin(data: MetricInput) -> float:
    return float(_clusters(data)["davies_bouldin"])


def _asi(data: MetricInput) -> float:
    return float(
        asi_score(
            data.validation_features,
            data.anomaly_scores,
            contamination=data.metric_tail_fraction,
        )
    )


def _asoi(data: MetricInput) -> float:
    return float(
        asoi_score(
            data.validation_features,
            data.anomaly_scores,
            contamination=data.metric_tail_fraction,
        )
    )


def _rtm(data: MetricInput) -> float:
    return float(relative_top_median_score(data.anomaly_scores))


def _eag(data: MetricInput) -> float:
    return float(expected_anomaly_gap_score(data.anomaly_scores))


def _npd(data: MetricInput) -> float:
    return float(
        normalized_pseudo_discrepancy_score(data.anomaly_scores, data.reference_scores)
    )


def _laplacian(data: MetricInput) -> float:
    neighbors = min(5, len(data.validation_features) - 1)
    return float(
        laplacian_score(
            data.validation_features, data.anomaly_scores, n_neighbors=neighbors
        )
    )


def _sireos(data: MetricInput) -> float:
    return float(sireos_score(data.validation_features, data.anomaly_scores))


def _ranking_stability(data: MetricInput) -> float:
    return float(
        ranking_stability_score(
            data.repeated_anomaly_scores,
            contamination=data.metric_tail_fraction,
        )
    )


def _top_k_stability(data: MetricInput) -> float:
    return float(
        top_k_stability_score(
            data.repeated_anomaly_scores,
            top_fraction=data.metric_tail_fraction,
        )
    )


def _excess_mass(data: MetricInput) -> float:
    return float(
        excess_mass_auc(
            data.anomaly_scores,
            data.reference_scores,
            support_volume=data.support_volume,
        )
    )


def _mass_volume(data: MetricInput) -> float:
    return float(
        mass_volume_auc(
            data.anomaly_scores,
            data.reference_scores,
            support_volume=data.support_volume,
        )
    )


def _score_matrix(inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
    return np.vstack([item.anomaly_scores for item in inputs])


def _model_centrality(inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
    return model_centrality_scores(_score_matrix(inputs))


def _average_rank_consensus(inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
    return average_rank_consensus_scores(_score_matrix(inputs))


def _hits(inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
    return hits_model_scores(_score_matrix(inputs))


def _ireos(inputs: Sequence[MetricInput]) -> NDArray[np.float64]:
    features = inputs[0].validation_features
    probabilities = np.vstack([item.outlier_probabilities for item in inputs])
    # One explicit range avoids candidate-specific automatic kernel widths.
    return ireos_scores(features, probabilities, gamma_max=1.0)


SCORE_SILHOUETTE = MetricSpec("score_silhouette", "maximize", _silhouette)
SCORE_CALINSKI_HARABASZ = MetricSpec(
    "score_calinski_harabasz", "maximize", _calinski_harabasz
)
SCORE_DAVIES_BOULDIN = MetricSpec("score_davies_bouldin", "minimize", _davies_bouldin)
ASI = MetricSpec("asi", "maximize", _asi)
ASOI = MetricSpec("asoi", "maximize", _asoi)
RTM = MetricSpec("rtm", "maximize", _rtm)
EAG = MetricSpec("eag", "maximize", _eag)
NPD = MetricSpec("npd", "maximize", _npd)
LAPLACIAN = MetricSpec("laplacian", "minimize", _laplacian)
SIREOS = MetricSpec("sireos", "minimize", _sireos)
RANKING_STABILITY = MetricSpec("ranking_stability", "maximize", _ranking_stability)
TOP_K_STABILITY = MetricSpec("top_k_stability", "maximize", _top_k_stability)
EXCESS_MASS = MetricSpec("excess_mass", "maximize", _excess_mass)
MASS_VOLUME = MetricSpec("mass_volume", "minimize", _mass_volume)
MODEL_CENTRALITY = MetricSpec(
    "model_centrality", "maximize", pool_compute=_model_centrality
)
AVERAGE_RANK_CONSENSUS = MetricSpec(
    "average_rank_consensus", "maximize", pool_compute=_average_rank_consensus
)
HITS = MetricSpec("hits", "maximize", pool_compute=_hits)
IREOS = MetricSpec("ireos", "maximize", pool_compute=_ireos)

REFERENCE_METRICS = (
    SCORE_SILHOUETTE,
    SCORE_CALINSKI_HARABASZ,
    SCORE_DAVIES_BOULDIN,
    ASI,
    ASOI,
    RTM,
    EAG,
    NPD,
    LAPLACIAN,
    MODEL_CENTRALITY,
    AVERAGE_RANK_CONSENSUS,
    HITS,
    RANKING_STABILITY,
    TOP_K_STABILITY,
    IREOS,
    SIREOS,
    EXCESS_MASS,
    MASS_VOLUME,
)


def load_metrics(import_paths: tuple[str, ...]) -> tuple[MetricSpec, ...]:
    """Load individual metrics or a tuple such as ``REFERENCE_METRICS``."""
    metrics: list[MetricSpec] = []
    for import_path in import_paths:
        module_name, separator, attribute = import_path.partition(":")
        if not separator:
            raise ValueError(f"metric path must use module:attribute: {import_path}")
        value = getattr(import_module(module_name), attribute)
        if isinstance(value, MetricSpec):
            metrics.append(value)
        elif isinstance(value, (tuple, list)) and all(
            isinstance(item, MetricSpec) for item in value
        ):
            metrics.extend(value)
        else:
            raise TypeError(f"{import_path} is not a MetricSpec or metric collection")
    names = [metric.name for metric in metrics]
    if len(names) != len(set(names)):
        raise ValueError("metric names must be unique")
    return tuple(metrics)
