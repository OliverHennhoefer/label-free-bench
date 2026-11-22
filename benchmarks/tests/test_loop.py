import json

import pytest

from benchmarks.loop import append, decide
from benchmarks.memory import check
from benchmarks.protocol import digest


@pytest.fixture
def lab(tmp_path):
    root = tmp_path / "research"
    (root / "experiments").mkdir(parents=True)
    (root / "ledger.jsonl").write_text("")
    return root


def experiment(root, slug, selector=None):
    folder = root / "experiments" / slug
    folder.mkdir()
    (folder / "prediction.md").write_text("Prediction.")
    if selector:
        (folder / "selector.py").write_text(selector)
    return folder


def grade(root, folder):
    row = {"run": "a" * 12, "split": "dev", "selectors": {}}
    row["selectors"]["selector@x"] = digest(folder / "selector.py")[:12]
    with (root / "ledger.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


def finish(root, slug, n):
    for role in ("proposer", "runner", "verifier"):
        append(root, role, slug, "done", None, f"{slug}-{role}-{n}")


def test_roles_follow_the_experiment(lab):
    assert decide(lab).role == "proposer"
    folder = experiment(lab, "e1", "def select(batch):\n    return None\n")
    append(lab, "proposer", "e1", "proposed", None, "s1")
    assert (decide(lab).role, decide(lab).slug) == ("runner", "e1")
    grade(lab, folder)
    assert (decide(lab).role, decide(lab).slug) == ("verifier", "e1")
    append(lab, "runner", "e1", "graded", None, "s2")
    append(lab, "verifier", "e1", "verified", None, "s3")
    (folder / "result.md").write_text("Result.")
    assert decide(lab).role == "observer"


def test_one_session_cannot_hold_two_roles(lab):
    experiment(lab, "e1")
    append(lab, "runner", "e1", "graded", None, "same")
    with pytest.raises(ValueError, match="fresh session"):
        append(lab, "verifier", "e1", "verified", None, "same")
    (lab / "experiments" / "e1" / "result.md").write_text("Result.")
    assert any("verifier session" in p for p in check(lab))


def test_cadence_gardening_and_halts(lab):
    (lab / "loop.toml").write_text(
        "[cadence]\nobserver_every = 0\nobserver_every_after = 0\ngardener_every = 2\n"
    )
    finish(lab, "e1", 1)
    assert decide(lab).role == "proposer"
    finish(lab, "e2", 2)
    assert decide(lab).role == "gardener"
    append(lab, "gardener", None, "gardened", None, "g")
    assert decide(lab).role == "proposer"
    (lab / "HANDOFF.md").write_text("**FLAG.** Too good to be true.\n")
    assert decide(lab).role == "halt"


def test_stuck_runner_goes_to_verifier_and_requests_wait(lab):
    experiment(lab, "e1")
    experiment(lab, "e2")
    (lab / "experiments" / "e2" / "heldout-request.md").write_text("Please.")
    append(lab, "runner", "e1", "crashed", None, "r1")
    append(lab, "runner", "e1", "crashed", None, "r2")
    assignment = decide(lab)
    assert (assignment.role, assignment.slug) == ("verifier", "e1")
    assert assignment.waiting and "e2" in assignment.waiting[0]


def test_observer_warmup_then_slower(lab):
    (lab / "loop.toml").write_text(
        "[cadence]\nobserver_warmup = 1\nobserver_every_after = 2\n"
    )
    finish(lab, "e1", 1)
    assert decide(lab).role == "observer"
    append(lab, "observer", None, "report", None, "o1")
    finish(lab, "e2", 2)
    assert decide(lab).role == "proposer"
    finish(lab, "e3", 3)
    assert decide(lab).role == "observer"
