"""Focused PyOD detector registry and versioned candidate design."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

SUPPORTED_MODELS = ("iforest", "lof", "pca", "ocsvm", "hbos")
CANDIDATE_POOL_VERSION = "1.1"


@dataclass(frozen=True)
class CandidateSpec:
    detector: str
    params: dict[str, Any]
    is_default: bool = False


def candidate_specs(
    models: tuple[str, ...],
    *,
    trials: int,
    fit_size: int,
    feature_count: int,
) -> list[CandidateSpec]:
    """Return a balanced deterministic pool with one default anchor per model."""
    unknown = sorted(set(models) - set(SUPPORTED_MODELS))
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    if len(set(models)) != len(models):
        raise ValueError("models must be unique")
    if trials % len(models):
        raise ValueError("trials must be divisible by the number of models")

    per_model = trials // len(models)
    by_model = {
        detector: _model_candidates(
            detector,
            count=per_model,
            fit_size=fit_size,
            feature_count=feature_count,
        )
        for detector in models
    }
    return [
        by_model[detector][round_index]
        for round_index in range(per_model)
        for detector in models
    ]


def _model_candidates(
    detector: str,
    *,
    count: int,
    fit_size: int,
    feature_count: int,
) -> list[CandidateSpec]:
    candidates = [CandidateSpec(detector, _default_params(detector), is_default=True)]
    if count == 1:
        return candidates
    if detector == "pca":
        return candidates + _pca_candidates(
            count - 1,
            fit_size=fit_size,
            feature_count=feature_count,
        )
    points = _stratified_points(detector, count - 1, dimensions=3)
    for point in points:
        candidates.append(
            CandidateSpec(
                detector,
                _params_from_point(
                    detector,
                    point,
                    fit_size=fit_size,
                    feature_count=feature_count,
                ),
            )
        )
    return candidates


def _pca_candidates(
    count: int, *, fit_size: int, feature_count: int
) -> list[CandidateSpec]:
    """Choose distinct PCA settings from a deterministic stratified grid."""
    upper = max(1, min(feature_count, fit_size) - 1)
    parameter_grid = [{"n_components": None, "weighted": False}]
    parameter_grid.extend(
        {"n_components": components, "weighted": weighted}
        for components in range(1, upper + 1)
        for weighted in (False, True)
    )
    if count > len(parameter_grid):
        raise ValueError(
            f"PCA needs {count} distinct non-default candidates, but only "
            f"{len(parameter_grid)} are available for {feature_count} features"
        )
    indices = np.linspace(0, len(parameter_grid) - 1, num=count, dtype=int)
    return [CandidateSpec("pca", parameter_grid[index]) for index in indices]


def _default_params(detector: str) -> dict[str, Any]:
    if detector == "iforest":
        return {"n_estimators": 100, "max_samples": "auto", "max_features": 1.0}
    if detector == "lof":
        return {"n_neighbors": 20, "p": 2}
    if detector == "pca":
        return {"n_components": None, "weighted": True}
    if detector == "ocsvm":
        return {"nu": 0.5, "gamma": "auto"}
    if detector == "hbos":
        return {"n_bins": 10, "alpha": 0.1, "tol": 0.5}
    raise ValueError(f"unknown detector {detector!r}")


def _params_from_point(
    detector: str,
    point: np.ndarray,
    *,
    fit_size: int,
    feature_count: int,
) -> dict[str, Any]:
    if detector == "iforest":
        estimators = (100, 200, 300, 400)
        return {
            "n_estimators": estimators[min(int(point[0] * 4), 3)],
            "max_samples": _linear(point[1], 0.5, 1.0),
            "max_features": _linear(point[2], 0.5, 1.0),
        }
    if detector == "lof":
        upper = max(5, min(40, fit_size - 1))
        return {
            "n_neighbors": _integer(point[0], 5, upper),
            "p": 1 if point[1] < 0.5 else 2,
        }
    if detector == "ocsvm":
        return {
            "nu": _log(point[0], 0.01, 0.5),
            "gamma": _log(point[1], 1e-4, 1.0),
        }
    if detector == "hbos":
        return {
            "n_bins": _integer(point[0], 5, 50),
            "alpha": _log(point[1], 0.01, 0.5),
            "tol": _linear(point[2], 0.1, 0.9),
        }
    raise ValueError(f"unknown detector {detector!r}")


def _stratified_points(name: str, count: int, *, dimensions: int) -> np.ndarray:
    """Small deterministic Latin-hypercube design independent of run seeds."""
    digest = hashlib.sha256(f"{CANDIDATE_POOL_VERSION}:{name}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    points = np.empty((count, dimensions), dtype=float)
    centers = (np.arange(count, dtype=float) + 0.5) / count
    for dimension in range(dimensions):
        points[:, dimension] = rng.permutation(centers)
    return points


def _linear(value: float, low: float, high: float) -> float:
    return float(low + value * (high - low))


def _log(value: float, low: float, high: float) -> float:
    return float(math.exp(math.log(low) + value * (math.log(high) - math.log(low))))


def _integer(value: float, low: int, high: int) -> int:
    return min(high, low + int(value * (high - low + 1)))


def build_detector(spec: CandidateSpec, *, seed: int, detector_contamination: float):
    if spec.detector == "iforest":
        return IForest(
            **spec.params,
            contamination=detector_contamination,
            random_state=seed,
            n_jobs=-1,
        )
    if spec.detector == "lof":
        return LOF(
            **spec.params,
            contamination=detector_contamination,
            n_jobs=-1,
        )
    if spec.detector == "pca":
        return PCA(
            **spec.params,
            contamination=detector_contamination,
            random_state=seed,
            standardization=False,
        )
    if spec.detector == "ocsvm":
        return OCSVM(
            **spec.params,
            contamination=detector_contamination,
            kernel="rbf",
            cache_size=512,
        )
    if spec.detector == "hbos":
        return HBOS(**spec.params, contamination=detector_contamination)
    raise ValueError(f"unknown detector {spec.detector!r}")


def fit_detector(detector, X: np.ndarray):
    """Fit a detector and conservatively stabilize rank-deficient PyOD PCA."""
    with np.errstate(divide="ignore", invalid="ignore"):
        detector.fit(X)
    scores = np.asarray(detector.decision_scores_, dtype=float)
    if isinstance(detector, PCA) and not np.isfinite(scores).all():
        weights = np.asarray(detector.selected_w_components_, dtype=float)
        positive = weights[np.isfinite(weights) & (weights > 0)]
        if positive.size == 0:
            raise ValueError("PCA produced no positive component weights")
        # PyOD's weighted PCA divides by explained-variance ratios. Exact
        # rank deficiency therefore yields infinities. A zero component is
        # conservatively assigned the smallest positive weight instead of
        # receiving unbounded influence.
        detector.selected_w_components_ = np.where(
            np.isfinite(weights) & (weights > 0), weights, positive.min()
        )
        detector.decision_scores_ = detector.decision_function(X)
        detector._process_decision_scores()
        scores = np.asarray(detector.decision_scores_, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("detector produced non-finite training scores")
    return detector
