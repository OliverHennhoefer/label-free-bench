"""Held-out looks, run only by the CI workflow that holds the grading secret.

The CI job builds the validation or test bank from a secret-keyed row sample,
runs the committed selectors on it, grades them and appends a signed ledger row.
Agents never see the held-out bank, its key or the secret. They see the ledger
row and, for validation, the rounded report.

Order matters inside the job. The secret is removed from the environment before
any selector runs, and the grading key is held in memory, not on disk, while
selector code executes.
"""

import hmac
import json
import os
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

from . import grade as grading
from .prepare import HELD_OUT, SALT_VARIABLE, prepare
from .protocol import digest, read_json
from .run import run

# Fixed so that a held-out look cannot be repeated with fresh seeds.
SEEDS = (0, 1, 2)
MAX_FINALISTS = 2
EXPERIMENTS = Path("research/experiments")
LOOKS = Path("research/looks")


def canonical(entry):
    body = {k: v for k, v in entry.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, allow_nan=False).encode()


def sign(entry, salt):
    return hmac.new(salt.encode(), canonical(entry), "sha256").hexdigest()


def verify_ledger(path, salt):
    """Return problems with held-out rows: unsigned, forged or edited."""
    problems = []
    if not Path(path).is_file():
        return problems
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line:
            continue
        entry = json.loads(line)
        if entry.get("split") not in HELD_OUT:
            continue
        expected = sign(entry, salt)
        if not hmac.compare_digest(entry.get("signature") or "", expected):
            problems.append(f"{path}:{number}: held-out row {entry.get('run')} fails")
    return problems


def verify(args):
    salt = os.environ.get(SALT_VARIABLE)
    if not salt:
        raise PermissionError(f"{SALT_VARIABLE} is missing. Run this in CI only.")
    problems = verify_ledger(args.ledger, salt)
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)
    print(f"Every held-out row in {args.ledger} carries a valid signature.")


def tracked(path):
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(path)],
        capture_output=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", str(path)], check=False
    )
    return result.returncode == 0 and clean.returncode == 0


def check_request(split, selectors, prediction, reference, ledger):
    """Refuse a look that the protocol does not allow, before anything is built.

    `reference` is a baseline name or the path of another committed selector.
    """
    if split not in HELD_OUT:
        raise ValueError("Held-out looks are for validation or test")
    if not selectors:
        raise ValueError("Name at least one selector")
    if len(set(selectors)) != len(selectors):
        raise ValueError("Name each selector once")
    root = EXPERIMENTS.resolve()
    reference_path = [Path(reference)] if str(reference).endswith(".py") else []
    for path in [*selectors, *reference_path]:
        resolved = path.resolve()
        if not resolved.is_relative_to(root) or resolved.suffix != ".py":
            raise ValueError(f"Selectors must be .py files under {EXPERIMENTS}/")
        if not tracked(path):
            raise ValueError(f"Commit the selector before a held-out look: {path}")
    grading.committed(prediction)
    text = prediction.read_text()
    unnamed = [str(p) for p in selectors if str(p) not in text]
    if unnamed:
        raise ValueError(
            "The committed prediction must name every selector it covers: "
            + ", ".join(unnamed)
        )
    if split == "test":
        if len(selectors) > MAX_FINALISTS:
            raise ValueError(f"Test is for at most {MAX_FINALISTS} finalists")
        if Path(reference) in selectors:
            raise ValueError("The test reference cannot also be a finalist")
        tested = set()
        if ledger.is_file():
            for line in ledger.read_text().splitlines():
                entry = json.loads(line) if line else {}
                if entry.get("split") == "test":
                    tested.update(entry.get("selectors", {}).values())
        again = [str(p) for p in selectors if digest(p)[:12] in tested]
        if again:
            raise ValueError("Each selector gets one test look: " + ", ".join(again))


def look(args):
    salt = os.environ.pop(SALT_VARIABLE, None)
    if not salt:
        raise PermissionError(f"{SALT_VARIABLE} is missing. Run this in CI only.")
    selectors = [Path(p) for p in args.selectors]
    reference_path = Path(args.reference) if args.reference.endswith(".py") else None
    check_request(args.split, selectors, args.prediction, args.reference, args.ledger)
    problems = verify_ledger(args.ledger, salt)
    if problems:
        raise ValueError("Ledger failed verification: " + "; ".join(problems))
    with TemporaryDirectory(prefix="heldout-") as scratch:
        scratch = Path(scratch)
        os.environ[SALT_VARIABLE] = salt
        try:
            prepare(
                Namespace(
                    raw_dir=args.raw_dir,
                    download=True,
                    split=args.split,
                    datasets=None,
                    seeds=list(SEEDS),
                    max_samples=args.max_samples,
                    extra_views=args.extra_views,
                    out=scratch / "bank",
                )
            )
        finally:
            os.environ.pop(SALT_VARIABLE, None)
        # Keep the key in memory while candidate code runs in this job.
        key_path = scratch / "bank" / "private"
        key = {p.name: p.read_bytes() for p in key_path.iterdir()}
        shutil.rmtree(key_path)
        if args.sandbox:
            # The unprivileged worker reads the public bank and selector snapshots.
            os.chmod(scratch, 0o755)
        public = scratch / "bank" / "public"
        submissions = [scratch / "baselines"]
        run(_run_args(public, None, submissions[0]))
        names = {}
        extra = (
            [reference_path]
            if reference_path and reference_path not in selectors
            else []
        )
        for index, path in enumerate([*selectors, *extra]):
            out = scratch / f"selector-{index}"
            run(_run_args(public, path, out, args.sandbox))
            first = next(iter(read_json(out / "submission.json")["tasks"].values()))
            (names[str(path)],) = first
            submissions.append(out)
        reference = names.get(str(reference_path), args.reference)
        os.chmod(scratch, 0o700)
        truth = scratch / "key"
        truth.mkdir(mode=0o700)
        for name, content in key.items():
            (truth / name).write_bytes(content)
        report_dir = scratch / "report"
        grading.grade(
            Namespace(
                truth=truth,
                submissions=submissions,
                target=args.target,
                reference=reference,
                out=report_dir,
                prediction=args.prediction,
                ledger=args.ledger,
                signer=lambda entry: sign(entry, salt),
                ci_run=args.ci_run,
                published=args.looks,
            )
        )
        run_id = json.loads(args.ledger.read_text().splitlines()[-1])["run"]
        args.looks.mkdir(parents=True, exist_ok=True)
        # Validation reports are rounded and test reports hold only decisions.
        shutil.copy(report_dir / "report.json", args.looks / f"{run_id}-report.json")
        if args.split == "test":
            # Win counts and p-values stay private. CI encrypts this file.
            (private,) = (truth / "looks").iterdir()
            shutil.copy(private, args.private_out)
    print(f"Held-out look {run_id} recorded")


def _run_args(bank, selector, out, sandbox=None):
    return Namespace(
        bank=bank,
        selector=selector,
        hypothesis=None,
        ireos_rows=None,
        timeout=120,
        out=out,
        sandbox=sandbox,
    )
