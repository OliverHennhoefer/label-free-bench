import numpy as np
import pytest

from label_free_bench.models import (
    CandidateSpec,
    build_detector,
    candidate_specs,
    fit_detector,
)


@pytest.mark.parametrize(
    ("name", "params"),
    [
        (
            "iforest",
            {"n_estimators": 100, "max_samples": 0.8, "max_features": 1.0},
        ),
        ("lof", {"n_neighbors": 10, "p": 2}),
        ("pca", {"n_components": 3, "weighted": True}),
        ("ocsvm", {"nu": 0.1, "gamma": 0.1}),
        ("hbos", {"n_bins": 10, "alpha": 0.1, "tol": 0.5}),
    ],
)
def test_publication_detector_supports_required_score_api(
    name: str, params: dict[str, object]
) -> None:
    rng = np.random.default_rng(4)
    fit = rng.normal(size=(80, 6))
    validation = rng.normal(size=(20, 6))
    detector = build_detector(
        CandidateSpec(name, params),
        seed=4,
        detector_contamination=0.1,
    )

    detector.fit(fit)
    scores = detector.decision_function(validation)
    probabilities = detector.predict_proba(validation)[:, 1]

    assert scores.shape == (20,)
    assert probabilities.shape == (20,)
    assert np.isfinite(scores).all()
    assert np.isfinite(probabilities).all()


def test_weighted_pca_default_is_finite_on_rank_deficient_data() -> None:
    rng = np.random.default_rng(7)
    base = rng.normal(size=(80, 3))
    rank_deficient = np.column_stack((base, base[:, 0], np.ones(len(base))))
    spec = candidate_specs(
        ("pca",), trials=1, fit_size=len(base), feature_count=5
    )[0]
    detector = build_detector(spec, seed=7, detector_contamination=0.1)

    fit_detector(detector, rank_deficient)
    scores = detector.decision_function(rank_deficient)
    probabilities = detector.predict_proba(rank_deficient)[:, 1]

    assert np.isfinite(scores).all()
    assert np.isfinite(probabilities).all()


def test_candidate_pool_is_balanced_deterministic_and_default_anchored() -> None:
    models = ("iforest", "lof", "pca", "ocsvm", "hbos")
    first = candidate_specs(models, trials=10, fit_size=100, feature_count=6)
    second = candidate_specs(models, trials=10, fit_size=100, feature_count=6)

    assert first == second
    assert [candidate.detector for candidate in first[:5]] == list(models)
    assert {name: sum(row.detector == name for row in first) for name in models} == {
        name: 2 for name in models
    }
    assert first[0].params["max_samples"] == "auto"
    assert first[1].params == {"n_neighbors": 20, "p": 2}
