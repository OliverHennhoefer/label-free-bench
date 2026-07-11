"""Terminal entry point."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from label_free_bench.config import load_config
from label_free_bench.data import fetch_adbench_data
from label_free_bench.metrics import load_metrics
from label_free_bench.pipeline import (
    compare_runs,
    evaluate_metrics_from_reference,
    run_benchmark,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="label-free-bench",
        description="Compare normal-only model-selection metrics.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate a run configuration")
    validate.add_argument("--config", type=Path, required=True)

    fetch = subparsers.add_parser(
        "fetch-data", help="download and checksum the benchmark ADBench datasets"
    )
    fetch.add_argument(
        "--destination",
        type=Path,
        default=Path("data/adbench/Classical"),
    )
    fetch.add_argument("--datasets", nargs="+", help="optional dataset names")

    run = subparsers.add_parser("run", help="run or resume a benchmark")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--suite",
        choices=("development", "confirmation", "all"),
        default="development",
        help="dataset suite; confirmation additionally requires --confirm",
    )
    run.add_argument(
        "--confirm",
        action="store_true",
        help="authorize one confirmatory run on the frozen confirmation suite",
    )
    run.add_argument("--trials", type=int, help="override trials per dataset and seed")
    run.add_argument("--seeds", type=int, nargs="+", help="override configured seeds")
    run.add_argument(
        "--metric",
        action="append",
        help="replace configured metrics with module:attribute (repeatable)",
    )
    run.add_argument(
        "--reference",
        type=Path,
        help="stored reference run to compare with after this run finishes",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "fetch-data":
        paths = fetch_adbench_data(
            args.destination,
            set(args.datasets) if args.datasets else None,
        )
        print(f"datasets: {len(paths)} verified in {args.destination}")
        return 0

    config = load_config(args.config)
    load_metrics(config.metrics)
    if args.command == "validate":
        print(f"valid: {args.config}")
        return 0

    if args.trials is not None:
        if args.trials < 1:
            raise SystemExit("--trials must be at least 1")
        config = replace(config, trials=args.trials)
    if args.seeds:
        config = replace(config, seeds=tuple(args.seeds))
    if args.metric:
        config = replace(config, metrics=tuple(args.metric))
        load_metrics(config.metrics)
    if args.reference:
        result = evaluate_metrics_from_reference(config, args.reference, args.output)
        result_path = compare_runs(args.reference, result.output_dir)
    else:
        if args.suite == "confirmation" and not args.confirm:
            raise SystemExit("confirmation suite requires --confirm")
        if args.confirm and args.suite != "confirmation":
            raise SystemExit("--confirm is valid only with --suite confirmation")
        result = run_benchmark(
            config,
            args.output,
            suite=args.suite,
            inference_mode=("confirmatory" if args.confirm else "exploratory"),
        )
        result_path = result.results_path
    print(f"results: {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
