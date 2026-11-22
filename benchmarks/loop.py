"""Decide which research role runs next, and log each role session.

The loop's skeleton is fixed here: role order, gardening and observer cadence,
fresh sessions per role, and a halt when a human is needed. What a role does
inside its session is left to the role. The repository is the only shared
state, so the same decision comes out of any checkout.

`next` reads the repository and prints one assignment. `log` appends one line
to `research/loop.jsonl` when a role session ends.
"""

import json
import os
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .memory import check as check_memory
from .protocol import digest

ROLES = ("proposer", "runner", "verifier", "gardener", "observer")
EXPERIMENT_ROLES = ("proposer", "runner", "verifier")
DEFAULTS = {
    "gardener_every": 10,
    "observer_every": 1,
    "observer_warmup": 5,
    "observer_every_after": 5,
    "max_runner_attempts": 2,
}


@dataclass
class Assignment:
    role: str
    slug: str | None = None
    reason: str = ""
    waiting: list[str] = field(default_factory=list)

    def render(self, root: Path) -> str:
        lines = [f"role: {self.role}"]
        if self.slug:
            lines.append(f"experiment: {root / 'experiments' / self.slug}")
        if self.role != "halt":
            lines.append(f"instructions: {root / 'roles' / f'{self.role}.md'}")
        lines.append(f"reason: {self.reason}")
        lines += [f"waiting on a human: {item}" for item in self.waiting]
        return "\n".join(lines)


def settings(root: Path) -> dict:
    path = root / "loop.toml"
    values = tomllib.loads(path.read_text())["cadence"] if path.is_file() else {}
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
    return {**DEFAULTS, **values}


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def selector_hashes(folder: Path) -> set[str]:
    return {digest(path)[:12] for path in folder.glob("*.py")}


def graded(folder: Path, ledger: list[dict]) -> list[dict]:
    """Development rows that graded a selector file from this experiment folder."""
    hashes = selector_hashes(folder)
    return [row for row in ledger if hashes & set(row.get("selectors", {}).values())]


def experiments(root: Path) -> list[Path]:
    folders = (root / "experiments").glob("*/prediction.md")
    return sorted((p.parent for p in folders), key=lambda p: p.name)


def iterations(log: list[dict]) -> list[dict]:
    """A completed iteration is one verifier session."""
    return [row for row in log if row["role"] == "verifier"]


def since(log: list[dict], role: str) -> int:
    """Iterations completed after the last session of this role."""
    last = max((i for i, row in enumerate(log) if row["role"] == role), default=-1)
    return len(iterations(log[last + 1 :]))


def observer_due(log: list[dict], cadence: dict) -> bool:
    done = len(iterations(log))
    every = (
        cadence["observer_every"]
        if done <= cadence["observer_warmup"]
        else cadence["observer_every_after"]
    )
    return every > 0 and done > 0 and since(log, "observer") >= every


def flags(root: Path) -> list[str]:
    """Reasons the loop must stop until a human acts."""
    found = []
    handoff = root / "HANDOFF.md"
    if handoff.is_file():
        first = next((l for l in handoff.read_text().splitlines() if l.strip()), "")
        if first.lstrip("*# ").upper().startswith("FLAG"):
            found.append(f"{handoff}: {first.strip()}")
    reports = sorted((root.parent / "observer").glob("[0-9]*.md"))
    lines = reports[-1].read_text().lower().splitlines() if reports else []
    if any(line.strip(" *").startswith("halt: yes") for line in lines):
        found.append(f"{reports[-1]} asks to halt the loop")
    return found


def decide(root: Path = Path("research")) -> Assignment:
    cadence = settings(root)
    log = read_jsonl(root / "loop.jsonl")
    ledger = [r for r in read_jsonl(root / "ledger.jsonl") if r["split"] == "dev"]
    if stop := flags(root):
        return Assignment("halt", reason="a human is needed", waiting=stop)
    waiting = [
        f"held-out look requested in {p}"
        for p in sorted((root / "experiments").glob("*/heldout-request.md"))
    ]
    if problems := check_memory(root):
        return Assignment("gardener", reason=problems[0], waiting=waiting)
    if observer_due(log, cadence):
        return Assignment("observer", reason="observer cadence", waiting=waiting)
    if since(log, "gardener") >= cadence["gardener_every"]:
        return Assignment(
            "gardener",
            reason=f"{cadence['gardener_every']} iterations since gardening",
            waiting=waiting,
        )
    open_ = [p for p in experiments(root) if not (p / "result.md").is_file()]
    for folder in open_:
        if graded(folder, ledger):
            return Assignment("verifier", folder.name, "graded, not verified", waiting)
    for folder in open_:
        attempts = sum(
            row["role"] == "runner" and row.get("slug") == folder.name for row in log
        )
        if attempts >= cadence["max_runner_attempts"]:
            reason = f"{attempts} runner sessions produced no grade"
            return Assignment("verifier", folder.name, reason, waiting)
    for folder in open_:
        if not (folder / "heldout-request.md").is_file():
            return Assignment("runner", folder.name, "prediction committed", waiting)
    return Assignment("proposer", reason="no experiment awaits a run", waiting=waiting)


def session_id() -> str | None:
    return os.environ.get("LAB_SESSION") or os.environ.get("CLAUDE_CODE_SESSION_ID")


def separation_problems(log: list[dict]) -> list[str]:
    """One session may not hold two experiment roles for the same experiment."""
    problems, seen = [], {}
    for row in log:
        if row["role"] not in EXPERIMENT_ROLES or not row.get("slug"):
            continue
        roles = seen.setdefault((row["slug"], row["session"]), set())
        if roles - {row["role"]}:
            problems.append(
                f"session {row['session']} acted as {sorted(roles | {row['role']})} "
                f"on {row['slug']}. Each role needs a fresh session."
            )
        roles.add(row["role"])
    return problems


def append(root: Path, role, slug, summary, friction, session) -> dict:
    if role not in ROLES:
        raise ValueError(f"Unknown role {role!r}")
    if role in ("runner", "verifier") and not slug:
        raise ValueError(f"A {role} session must name its experiment")
    if not session:
        raise ValueError("No session ID. Set LAB_SESSION or pass --session.")
    if len(summary.split()) > 60 or len((friction or "").split()) > 60:
        raise ValueError("Keep the summary and friction under 60 words each")
    path = root / "loop.jsonl"
    entry = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "role": role,
        "slug": slug,
        "session": session,
        "summary": summary,
        "friction": friction,
    }
    if problems := separation_problems([*read_jsonl(path), entry]):
        raise ValueError(problems[-1])
    with path.open("a") as stream:
        stream.write(json.dumps(entry) + "\n")
    return entry


def next_role(args):
    print(decide(args.root).render(args.root))


def log_role(args):
    entry = append(
        args.root,
        args.role,
        args.slug,
        args.summary,
        args.friction,
        args.session or session_id(),
    )
    print(f"Logged {entry['role']} session {entry['session']} to loop.jsonl")
