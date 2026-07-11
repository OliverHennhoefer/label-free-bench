import csv
from pathlib import Path

import pytest

from label_free_bench.cli import main


def test_metric_override_uses_one_config_for_custom_runs(tmp_path: Path) -> None:
    config = tmp_path / "synthetic.yaml"
    config.write_text(
        "datasets: [{name: synthetic}]\n"
        "seeds: [1]\n"
        "models: [iforest, lof, pca, ocsvm, hbos]\n"
        "metrics: [label_free_bench.metrics:REFERENCE_METRICS]\n"
        "trials: 5\n"
    )
    output = tmp_path / "run"

    result = main(
        [
            "run",
            "--config",
            str(config),
            "--metric",
            "label_free_bench.user_metrics.metric:METRIC",
            "--output",
            str(output),
        ]
    )

    with (output / "artifacts" / "candidates.csv").open(newline="") as handle:
        fields = csv.DictReader(handle).fieldnames
    assert result == 0
    assert fields is not None
    assert "metric_user_tail_separation" in fields
    assert "metric_eag" not in fields
    assert (output / "results.csv").exists()
    assert {path.name for path in output.iterdir()} == {"results.csv", "artifacts"}


def test_confirmation_suite_requires_explicit_confirmation(tmp_path: Path) -> None:
    config = tmp_path / "confirmation.yaml"
    config.write_text(
        "datasets: [{name: synthetic, suite: confirmation}]\n"
        "seeds: [1]\n"
        "models: [iforest, lof, pca, ocsvm, hbos]\n"
        "metrics: [label_free_bench.metrics:EAG]\n"
        "trials: 5\n"
    )

    with pytest.raises(SystemExit, match="requires --confirm"):
        main(
            [
                "run",
                "--config",
                str(config),
                "--suite",
                "confirmation",
                "--output",
                str(tmp_path / "run"),
            ]
        )
