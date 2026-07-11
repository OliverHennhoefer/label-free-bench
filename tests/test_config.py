from pathlib import Path

import pytest
import yaml

from label_free_bench.config import load_config


def test_light_config_loads() -> None:
    config = load_config(Path("configs/light.yaml"))

    assert len(config.datasets) == 7
    assert config.seeds == (1,)
    assert config.models == ("iforest", "lof", "pca", "ocsvm", "hbos")
    assert config.trials == 10
    assert config.reference_size == 256
    assert config.stability_repeats == 3
    assert config.detector_contamination == 0.1
    assert config.metric_tail_fraction == 0.1
    assert config.controlled_prevalences == (0.01, 0.05, 0.1)
    assert {dataset.suite for dataset in config.datasets} == {"development"}


def test_benchmark_yaml_contains_only_pillar_parameters() -> None:
    raw = yaml.safe_load(Path("configs/benchmark.yaml").read_text())

    assert set(raw) == {"datasets", "seeds", "models", "metrics", "trials"}
    assert len(raw["datasets"]) == 15
    assert len(raw["seeds"]) == 10
    assert raw["trials"] == 60
    assert sum(item["suite"] == "development" for item in raw["datasets"]) == 7
    assert sum(item["suite"] == "confirmation" for item in raw["datasets"]) == 8


def test_detector_and_metric_fractions_are_separate(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        "datasets: [{name: synthetic}]\n"
        "seeds: [1]\n"
        "models: [iforest]\n"
        "metrics: [label_free_bench.metrics:EAG]\n"
    )
    with benchmark.open("a") as handle:
        handle.write("detector_contamination: 0.03\nmetric_tail_fraction: 0.2\n")

    config = load_config(benchmark)

    assert config.detector_contamination == 0.03
    assert config.metric_tail_fraction == 0.2


def test_config_rejects_empty_required_list(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("datasets: []\nseeds: [1]\nmodels: [iforest]\nmetrics: [x:y]\n")

    with pytest.raises(ValueError, match="datasets"):
        load_config(path)
