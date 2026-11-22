"""Keep the research memory small, linked and checkable.

Every number in a claim must trace to a ledger run. Size limits force entries to
stay atomic, so the index can be read in one pass and gardening has a trigger.
"""

import json
import re
from pathlib import Path

LIMITS = {  # path pattern: (unit, maximum)
    "README.md": ("lines", 150),
    "HANDOFF.md": ("lines", 40),
    "GARDENING.md": ("lines", 40),
    "claims/*.md": ("words", 150),
    "experiments/*/prediction.md": ("words", 250),
    "experiments/*/result.md": ("words", 350),
    "literature/*.md": ("words", 200),
    "literature/QUEUE.md": ("words", 400),
    "../observer/*.md": ("words", 300),
}
STATUSES = ("open", "supported", "refuted", "superseded by C-")
RUN_ID = re.compile(r"\b[0-9a-f]{12}\b")
NEGATIVE_ROW_WORDS = 60


def ledger_runs(root: Path) -> set[str]:
    path = root / "ledger.jsonl"
    if not path.is_file():
        return set()
    return {json.loads(line)["run"] for line in path.read_text().splitlines() if line}


def size(path: Path, unit: str) -> int:
    text = path.read_text()
    return len(text.splitlines()) if unit == "lines" else len(text.split())


def check(root: Path = Path("research")) -> list[str]:
    problems = []
    for pattern, (unit, maximum) in LIMITS.items():
        for path in sorted(root.glob(pattern)):
            if pattern == "literature/*.md" and path.name == "QUEUE.md":
                continue
            if (n := size(path, unit)) > maximum:
                problems.append(
                    f"{path}: {n} {unit}, limit {maximum}. Split or garden it."
                )
    runs = ledger_runs(root)
    for path in sorted(root.glob("claims/*.md")):
        text = path.read_text()
        if not re.match(rf"# {re.escape(path.stem)}: \S", text):
            problems.append(f"{path}: first line must be '# {path.stem}: <claim>'")
        fields = dict(
            line.split(":", 1) for line in text.splitlines()[1:] if ":" in line
        )
        status = fields.get("status", "").strip()
        if not status.startswith(STATUSES):
            problems.append(f"{path}: status must be one of {', '.join(STATUSES)}")
        evidence = RUN_ID.findall(fields.get("evidence", ""))
        if status != "open" and not evidence:
            problems.append(f"{path}: a decided claim needs ledger run IDs as evidence")
        for run in evidence:
            if run not in runs:
                problems.append(f"{path}: evidence {run} is not in the ledger")
    negative = root / "negative.md"
    if negative.is_file():
        for number, line in enumerate(negative.read_text().splitlines(), 1):
            if line.startswith("| N-") and len(line.split()) > NEGATIVE_ROW_WORDS:
                problems.append(
                    f"{negative}:{number}: row over {NEGATIVE_ROW_WORDS} words"
                )
    problems += loop_problems(root)
    return problems


def loop_problems(root: Path) -> list[str]:
    """Role separation: a verified experiment needs its own verifier session."""
    from .loop import read_jsonl, separation_problems

    log = read_jsonl(root / "loop.jsonl")
    problems = separation_problems(log)
    for result in sorted(root.glob("experiments/*/result.md")):
        slug = result.parent.name
        rows = [row for row in log if row.get("slug") == slug]
        if rows and not any(row["role"] == "verifier" for row in rows):
            problems.append(f"{result}: written without a logged verifier session")
    return problems


def main(args):
    problems = check(args.root)
    for problem in problems:
        print(problem)
    if problems:
        raise SystemExit(1)
    print(f"Research memory in {args.root} passes its checks.")
