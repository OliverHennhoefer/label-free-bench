"""Small, explicit configuration model for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    path: Path | None = None
    suite: Literal["development", "confirmation"] = "development"


@dataclass(frozen=True)
class RunConfig:
    datasets: tuple[DatasetConfig, ...]
    seeds: tuple[int, ...]
    models: tuple[str, ...]
    metrics: tuple[str, ...]
    trials: int
    test_fraction: float
    validation_fraction: float
    detector_contamination: float
    metric_tail_fraction: float
    reference_size: int
    stability_repeats: int
    max_normal_samples: int | None
    controlled_prevalences: tuple[float, ...]
    data_dir: Path | None = None


def load_config(path: str | Path) -> RunConfig:
    """Load and validate a benchmark configuration."""
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")

    datasets_raw = _required_list(raw, "datasets")
    datasets: list[DatasetConfig] = []
    for entry in datasets_raw:
        if not isinstance(entry, dict) or not entry.get("name"):
            raise ValueError("each dataset must be a mapping with a name")
        dataset_path = entry.get("path")
        suite = str(entry.get("suite", "development"))
        if suite not in {"development", "confirmation"}:
            raise ValueError("dataset suite must be development or confirmation")
        datasets.append(
            DatasetConfig(
                name=str(entry["name"]),
                path=_resolve(config_path, dataset_path) if dataset_path else None,
                suite=suite,
            )
        )

    data_dir = raw.get("data_dir", "../data/adbench/Classical")
    if "assumed_contamination" in raw:
        raise ValueError(
            "assumed_contamination was replaced by detector_contamination and "
            "metric_tail_fraction"
        )
    if "test_normal_fraction" in raw:
        raise ValueError("test_normal_fraction was replaced by test_fraction")
    max_normal_samples = raw.get("max_normal_samples", 500)
    config = RunConfig(
        datasets=tuple(datasets),
        seeds=tuple(int(value) for value in _required_list(raw, "seeds")),
        models=tuple(str(value) for value in _required_list(raw, "models")),
        metrics=tuple(str(value) for value in _required_list(raw, "metrics")),
        trials=int(raw.get("trials", 20)),
        test_fraction=float(raw.get("test_fraction", 0.5)),
        validation_fraction=float(raw.get("validation_fraction", 0.25)),
        detector_contamination=float(raw.get("detector_contamination", 0.1)),
        metric_tail_fraction=float(raw.get("metric_tail_fraction", 0.1)),
        reference_size=int(raw.get("reference_size", 256)),
        stability_repeats=int(raw.get("stability_repeats", 3)),
        max_normal_samples=(
            int(max_normal_samples) if max_normal_samples is not None else None
        ),
        controlled_prevalences=tuple(
            float(value)
            for value in raw.get("controlled_prevalences", [0.01, 0.05, 0.1])
        ),
        data_dir=_resolve(config_path, data_dir) if data_dir else None,
    )
    validate_config(config)
    return config


def validate_config(config: RunConfig) -> None:
    from label_free_bench.models import SUPPORTED_MODELS

    if (
        not config.datasets
        or not config.seeds
        or not config.models
        or not config.metrics
    ):
        raise ValueError("datasets, seeds, models, and metrics cannot be empty")
    if config.trials < 1:
        raise ValueError("trials must be at least 1")
    for name, values in (
        ("datasets", tuple(dataset.name for dataset in config.datasets)),
        ("seeds", config.seeds),
        ("models", config.models),
    ):
        if len(set(values)) != len(values):
            raise ValueError(f"{name} must be unique")
    unknown_models = sorted(set(config.models) - set(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(f"unknown models: {', '.join(unknown_models)}")
    if config.reference_size < 10:
        raise ValueError("reference_size must be at least 10")
    if config.stability_repeats < 2:
        raise ValueError("stability_repeats must be at least 2")
    if config.max_normal_samples is not None and config.max_normal_samples < 20:
        raise ValueError("max_normal_samples must be at least 20")
    if config.trials % len(config.models):
        raise ValueError("trials must be divisible by the number of models")
    if not config.controlled_prevalences:
        raise ValueError("controlled_prevalences cannot be empty")
    if len(set(config.controlled_prevalences)) != len(config.controlled_prevalences):
        raise ValueError("controlled_prevalences must be unique")
    for name, value in (
        ("test_fraction", config.test_fraction),
        ("validation_fraction", config.validation_fraction),
        ("detector_contamination", config.detector_contamination),
        ("metric_tail_fraction", config.metric_tail_fraction),
        *(
            ("controlled_prevalence", value)
            for value in config.controlled_prevalences
        ),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between 0 and 1")


def config_dict(config: RunConfig) -> dict[str, Any]:
    """Return a stable JSON-compatible configuration mapping."""
    return {
        "datasets": [
            {
                "name": item.name,
                "path": str(item.path) if item.path else None,
                "suite": item.suite,
            }
            for item in config.datasets
        ],
        "seeds": list(config.seeds),
        "models": list(config.models),
        "metrics": list(config.metrics),
        "trials": config.trials,
        "test_fraction": config.test_fraction,
        "validation_fraction": config.validation_fraction,
        "detector_contamination": config.detector_contamination,
        "metric_tail_fraction": config.metric_tail_fraction,
        "reference_size": config.reference_size,
        "stability_repeats": config.stability_repeats,
        "max_normal_samples": config.max_normal_samples,
        "controlled_prevalences": list(config.controlled_prevalences),
        "data_dir": str(config.data_dir) if config.data_dir else None,
    }


def experiment_dict(config: RunConfig) -> dict[str, Any]:
    """Return settings that must match when comparing two metric runs."""
    values = config_dict(config)
    values.pop("metrics")
    values.pop("data_dir")
    values["datasets"] = [
        {"name": item["name"], "suite": item["suite"]}
        for item in values["datasets"]
    ]
    return values


def _required_list(raw: dict[str, Any], name: str) -> list[Any]:
    value = raw.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    return value


def _resolve(config_path: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()
