import argparse
from pathlib import Path

from .protocol import TARGETS


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate label-free selectors for batch anomaly detection."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser(
        "prepare",
        help="Build separate public and private artifacts from pinned ADBench files",
    )
    prep.add_argument("--raw-dir", type=Path, required=True)
    prep.add_argument(
        "--download",
        action="store_true",
        help="Fetch missing files from the pinned ADBench revision",
    )
    prep.add_argument("--split", choices=("dev", "validation", "test"), default="dev")
    prep.add_argument(
        "--datasets", nargs="+", help="Optional ADBench stems within the selected split"
    )
    prep.add_argument("--seeds", type=int, nargs="+", default=[0])
    prep.add_argument("--max-samples", type=int, default=2000)
    prep.add_argument(
        "--extra-views",
        action="store_true",
        help="Cache refits and held-out reference scores for NPD, EM/MV and stability",
    )
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument(
        "--key-dir",
        type=Path,
        help="Move the private grading key here, outside the agent's workspace",
    )
    run = commands.add_parser(
        "run", help="Run the library baselines or one selector against a public bank"
    )
    run.add_argument("--bank", type=Path, required=True)
    run.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Wall-clock seconds per custom selector per batch, including startup",
    )
    run.add_argument(
        "--selector", type=Path, help="Standalone Python module exporting select(batch)"
    )
    run.add_argument(
        "--hypothesis", help="Prediction or rationale to record with this submission"
    )
    run.add_argument(
        "--ireos-rows",
        type=int,
        help="Also evaluate shared-range IREOS on up to this many rows",
    )
    run.add_argument("--out", type=Path, required=True)
    grade = commands.add_parser(
        "grade", help="Read frozen submissions and the private grading key"
    )
    grade.add_argument("--truth", type=Path, required=True)
    grade.add_argument("--submissions", type=Path, nargs="+", required=True)
    grade.add_argument(
        "--target",
        choices=TARGETS,
        default="ap",
        help="Supervised score whose detector choice the normal-only metric aims to match",
    )
    grade.add_argument(
        "--reference",
        default="iforest_default",
        help="Reference selector for paired score differences",
    )
    grade.add_argument("--out", type=Path, required=True)
    grade.add_argument(
        "--prediction",
        type=Path,
        help="Committed prediction file. Required for validation and test looks",
    )
    grade.add_argument(
        "--ledger",
        type=Path,
        default=Path("research/ledger.jsonl"),
        help="Append-only record of every look",
    )
    held = commands.add_parser(
        "heldout",
        help="CI only: build a held-out bank, run committed selectors, grade, sign",
    )
    held.add_argument("--split", choices=("validation", "test"), required=True)
    held.add_argument("--selectors", type=Path, nargs="+", required=True)
    held.add_argument("--prediction", type=Path, required=True)
    held.add_argument("--target", choices=TARGETS, default="ap")
    held.add_argument("--reference", default="iforest_default")
    held.add_argument("--extra-views", action="store_true")
    held.add_argument("--max-samples", type=int, default=2000)
    held.add_argument("--raw-dir", type=Path, default=Path(".bench-runs/raw"))
    held.add_argument("--ledger", type=Path, default=Path("research/ledger.jsonl"))
    held.add_argument("--looks", type=Path, default=Path("research/looks"))
    held.add_argument(
        "--private-out",
        type=Path,
        default=Path(".bench-runs/private-look.json"),
        help="Test win counts for the grader's eyes only. CI encrypts this file",
    )
    held.add_argument("--ci-run", help="URL of the CI run recorded in the ledger")
    held.add_argument(
        "--sandbox",
        help="Command prefix that runs each selector without privileges or network",
    )
    verify = commands.add_parser(
        "verify-ledger",
        help="CI only: check every held-out ledger row's signature",
    )
    verify.add_argument("--ledger", type=Path, default=Path("research/ledger.jsonl"))
    memory = commands.add_parser(
        "check-memory", help="Check research memory size limits and ledger links"
    )
    memory.add_argument("--root", type=Path, default=Path("research"))
    args = parser.parse_args()
    if args.command == "prepare":
        from .prepare import prepare as action
    elif args.command == "run":
        from .run import run as action
    elif args.command == "heldout":
        from .heldout import look as action
    elif args.command == "verify-ledger":
        from .heldout import verify as action
    elif args.command == "check-memory":
        from .memory import main as action
    else:
        from .grade import grade as action
    action(args)


if __name__ == "__main__":
    main()
