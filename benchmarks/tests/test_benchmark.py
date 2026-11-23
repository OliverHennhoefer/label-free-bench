import csv
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from benchmarks import grade as grade_module
from benchmarks import prepare
from benchmarks.grade import evaluate_choice, grade, group_means
from benchmarks.protocol import (
    DEFAULT,
    TARGETS,
    VERSION,
    digest,
    fingerprint,
    read_json,
    top_membership,
    write_json,
)
from benchmarks.run import run


def test_manifest_keeps_variants_and_public_examples_together():
    rows = [
        r
        for split in ("dev", "validation", "test")
        for r in prepare.datasets(split, [])
    ]
    assert len(rows) == len({r["dataset"] for r in rows}) == 47
    for source in {r["source"] for r in rows}:
        assert len({r["split"] for r in rows if r["source"] == source}) == 1
    examples = {
        "2_annthyroid",
        "4_breastw",
        "6_cardio",
        "29_Pima",
        "38_thyroid",
        "42_WBC",
    }
    assert examples <= {r["dataset"] for r in prepare.datasets("dev", [])}
    with pytest.raises(ValueError, match="Unknown datasets"):
        prepare.datasets("test", ["6_cardio"])


def test_private_targets_and_threshold_ties():
    labels = np.array([0, 0, 0, 1])
    scores = np.array([[1, 1, 1, 1], [0, 0, 1, 1]])
    values = prepare.oracle(labels, scores, f1_thresholds=[0.5, 0.5])
    assert values["ap"] == [0.25, 0.5]
    assert values["auroc"] == pytest.approx([0.5, 5 / 6])
    assert values["precision_at_budget"] == [0.25, 0.5]
    assert values["pr_auc_trapezoid"] != values["ap"]
    np.testing.assert_allclose(top_membership(scores[1], 1), [0, 0, 0.5, 0.5])
    assert values["f1"] == [0.4, 2 / 3]
    assert prepare.oracle(np.zeros(4), scores, f1_thresholds=[0, 0]) is None


def test_hierarchical_average():
    rows = [{"source": "a", "dataset": "x", "value": v} for v in [0.0, 0.0, 0.0]] + [
        {"source": "a", "dataset": "y", "value": 1.0},
        {"source": "b", "dataset": "z", "value": 1.0},
    ]
    np.testing.assert_array_equal(group_means(rows, "value"), [0.5, 1.0])


def test_same_detector_choice_can_match_roc_auc_and_miss_f1():
    y = np.array([0, 0, 1, 1])
    scores = np.array([[0, 1, 2, 3], [-1, 3, 2, 1]])
    outcomes = prepare.oracle(y, scores, f1_thresholds=[3, 0])
    assert outcomes["auroc"] == [1.0, 0.5]
    assert outcomes["f1"] == pytest.approx([0, 0.8])  # Equal to the cutoff is normal.
    choice = evaluate_choice({"status": "ok", "ranks": [1, 2]}, outcomes, ["A", "B"], 0)
    assert choice["auroc_regret"] == 0 and choice["auroc_best_pick_rate"] == 1
    assert (
        choice["f1_regret"] == pytest.approx(0.8) and choice["f1_best_pick_rate"] == 0
    )


@pytest.fixture
def bank(tmp_path):
    public, private = tmp_path / "bank", tmp_path / "key"
    public.mkdir()
    private.mkdir()
    scores = np.array([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    file = public / "task.npz"
    np.savez(file, X=np.arange(8).reshape(4, 2), scores=scores)
    manifest = {
        "version": VERSION,
        "families": ["a", "b"],
        "tasks": {"task": digest(file)},
    }
    bank_id = fingerprint(manifest)
    write_json(public / "bank.json", {**manifest, "bank_id": bank_id})
    task = {
        "dataset": "x",
        "source": "x",
        "seed": 0,
        "n": 4,
        "features": 2,
        "outcomes": prepare.oracle(
            np.array([0, 0, 0, 1]), scores, f1_thresholds=[0.5, 0.5]
        ),
    }
    write_json(
        private / "truth.json",
        {
            "split": "dev",
            "bank_id": bank_id,
            "candidates": [DEFAULT, "other"],
            "tasks": {"task": task},
        },
    )
    return public, private


def test_freeze_grade_and_failures(tmp_path, bank, monkeypatch):
    public, private = bank
    trials = []
    for name, expression in [
        ("flat", "[2,2]"),
        ("bad", "[float('nan'), 1]"),
        ("abstain", "None"),
    ]:
        module = tmp_path / f"{name}.py"
        module.write_text(
            f'"""Example hypothesis."""\ndef select(batch):\n'
            f'    assert not hasattr(batch, "y") and not batch.scores.flags.writeable\n'
            f"    return {expression}\n"
        )
        out = tmp_path / name
        args = Namespace(
            bank=public,
            out=out,
            selector=module,
            hypothesis=None,
            ireos_rows=None,
            timeout=30,
        )
        run(args)
        assert (out / "selector.py").read_bytes() == module.read_bytes()
        trials.append(out)
        with pytest.raises(FileExistsError):
            run(args)
    args = Namespace(
        truth=private,
        submissions=trials,
        out=tmp_path / "report",
        reference="iforest_default",
    )
    grade(args)
    report = read_json(args.out / "report.json")
    assert report["target"] == "ap"
    assert report["supervised_winners"] == {"task": [DEFAULT]}
    flat, bad, abstain = (
        next(k for k in report["summary"] if k.startswith(n + "@"))
        for n in ("flat", "bad", "abstain")
    )
    assert report["summary"][flat]["ap"] == report["summary"]["uniform_model"]["ap"]
    assert report["summary"][bad]["status_counts"]["failed"] == 1
    assert report["summary"][abstain]["status_counts"]["abstained"] == 1
    assert report["summary"][bad]["ap"] == report["summary"]["iforest_default"]["ap"]
    assert report["summary"][flat]["delta_ap_ci95"] is None
    assert report["by_task"][flat][0]["selected"] == [DEFAULT, "other"]
    assert report["summary"][flat]["ap_best_pick_rate"] == 0.5
    dev_ledger = tmp_path / "dev-ledger.jsonl"
    grade(Namespace(**{**vars(args), "out": tmp_path / "dev-2"}, ledger=dev_ledger))
    (row,) = [json.loads(line) for line in dev_ledger.read_text().splitlines()]
    assert row["results"][flat]["picks"] == {DEFAULT: 0.5, "other": 0.5}
    assert (
        row["results"][flat]["sources"] == 1 and "delta_ap_ci95" in row["results"][flat]
    )
    with pytest.raises(ValueError, match="Target 'brier' is unavailable"):
        grade(Namespace(**vars(args), target="brier"))

    key = read_json(private / "truth.json")
    key["split"] = "test"
    write_json(private / "truth.json", key)
    args.out = tmp_path / "holdout-report"
    with pytest.raises(ValueError, match="needs --prediction"):
        grade(args)
    prediction = tmp_path / "prediction.md"
    prediction.write_text("Expect no gain.")
    with pytest.raises(ValueError, match="Commit the prediction"):
        grade(Namespace(**vars(args), prediction=prediction))
    monkeypatch.setattr(grade_module, "committed", lambda path: "c0ffee")
    ledger = tmp_path / "ledger.jsonl"
    grade(Namespace(**vars(args), prediction=prediction, ledger=ledger))
    hidden_report = read_json(args.out / "report.json")
    assert "summary" not in hidden_report and "by_task" not in hidden_report
    assert "supervised_winners" not in hidden_report
    assert hidden_report["decisions"][flat] == "hold"
    (entry,) = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert entry["split"] == "test" and entry["prediction_commit"] == "c0ffee"
    assert entry["selectors"][flat] == flat.split("@")[1]
    assert entry["results"] == hidden_report["decisions"]
    (look,) = (private / "looks").iterdir()
    assert read_json(look)[flat]["source_wins"] == 0

    key["split"] = "validation"
    write_json(private / "truth.json", key)
    args.out = tmp_path / "validation-report"
    grade(Namespace(**vars(args), prediction=prediction, ledger=ledger))
    summary = read_json(args.out / "report.json")["summary"]

    def leaves(value):
        if isinstance(value, dict):
            return [x for v in value.values() for x in leaves(v)]
        if isinstance(value, list):
            return [x for v in value for x in leaves(v)]
        return [value]

    assert all(round(v, 2) == v for v in leaves(summary) if isinstance(v, float))
    assert grade_module.rounded({"ci": [0.12345, 0.98765]}) == {"ci": [0.12, 0.99]}
    assert len(ledger.read_text().splitlines()) == 2
    key["bank_id"] = "wrong"
    write_json(private / "truth.json", key)
    with pytest.raises(ValueError, match="different benchmark bank"):
        grade(Namespace(**vars(args), prediction=prediction))


def test_corrupt_bank_and_incomplete_submission(tmp_path, bank):
    public, private = bank
    plugin = tmp_path / "metric.py"
    plugin.write_text("def select(batch): return [1,2]")
    args = Namespace(
        bank=public,
        out=tmp_path / "out",
        selector=plugin,
        hypothesis="test",
        ireos_rows=None,
        timeout=30,
    )
    run(args)
    file = args.out / "submission.json"
    trial = read_json(file)
    trial["tasks"] = {}
    write_json(file, trial)
    with pytest.raises(ValueError, match="Incomplete"):
        grade(
            Namespace(
                truth=private,
                submissions=[args.out],
                out=tmp_path / "scores",
                reference="iforest_default",
            )
        )
    (public / "task.npz").write_bytes(b"broken")
    args.out = tmp_path / "second"
    with pytest.raises(ValueError, match="checksum"):
        run(args)
    assert not args.out.exists()


def test_hypothesis_has_time_budget(tmp_path, bank):
    public, _ = bank
    plugin = tmp_path / "slow.py"
    plugin.write_text("from time import sleep\ndef select(batch): sleep(60)")
    out = tmp_path / "timed"
    run(
        Namespace(
            bank=public,
            out=out,
            selector=plugin,
            hypothesis=None,
            ireos_rows=None,
            timeout=0.2,
        )
    )
    (result,) = read_json(out / "submission.json")["tasks"]["task"].values()
    assert result["status"] == "failed"
    assert "per-task budget" in result["reason"]


def test_pinned_download_rejects_wrong_blob(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    fake = tmp_path / "response"
    fake.write_bytes(b"wrong content")
    monkeypatch.setattr(prepare, "urlopen", lambda *a, **kw: fake.open("rb"))
    row = {"dataset": "fake", "git_blob": "0" * 40}
    with pytest.raises(ValueError, match="does not match"):
        prepare.verified_file(raw, row, True)
    assert list(raw.iterdir()) == []


def test_changing_private_measurements_changes_only_the_grades(monkeypatch):
    class FirstFeature:
        def fit(self, X):
            self.mean = X.mean(axis=0)
            return self

        def decision_function(self, X):
            return (X - self.mean)[:, 0]

    monkeypatch.setattr(prepare, "portfolio", lambda seed: {"probe": FirstFeature()})
    X = np.random.default_rng(6).normal(size=(192, 4))
    y = (np.arange(len(X)) % 4 == 0).astype(int)
    fit, selection, test = prepare.normal_partitions(y)
    assert not y[fit].any() and not y[selection].any()
    assert len(np.unique(np.r_[fit, selection, test])) == len(fit) + len(
        selection
    ) + len(test)
    assert set(y[test]) == {0, 1}
    X[y == 1, 0] = 100
    X[:, -1] = 0
    changed = X.copy()
    changed[test] += (
        10  # Even private normal observations must be invisible at selection.
    )
    changed[y == 1] = -100
    public, labels, hidden = prepare.build_task(X, y, 6, True)
    public_again, labels_again, hidden_again = prepare.build_task(changed, y, 6, True)
    assert public["X"].shape == (len(selection), 3)
    for key in public:
        np.testing.assert_array_equal(public[key], public_again[key])
    cutoff = public["scores"].max(axis=1)
    assert prepare.oracle(labels, hidden, f1_thresholds=cutoff)["auroc"] == [1.0]
    assert prepare.oracle(labels_again, hidden_again, f1_thresholds=cutoff)[
        "auroc"
    ] == [0.0]


def test_prepare_and_baselines_on_pyod(tmp_path, monkeypatch):
    pytest.importorskip("pyod")
    import hashlib

    from benchmarks.protocol import read_batch

    path = tmp_path / "fixture.npz"
    rng = np.random.default_rng(12)
    X = rng.normal(size=(280, 6))
    X[:3, 2] = np.nan
    X[:, 5] = 1
    y = (np.arange(len(X)) % 7 == 0).astype(int)
    np.savez(path, X=X, y=y)
    data = path.read_bytes()
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    manifest = tmp_path / "datasets.csv"
    with manifest.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(
            [
                ["dataset", "source", "split", "git_blob"],
                ["fixture", "group", "dev", blob],
            ]
        )
    monkeypatch.setattr(prepare, "MANIFEST", manifest)
    args = Namespace(
        raw_dir=tmp_path,
        datasets=["fixture"],
        split="dev",
        seeds=[5],
        max_samples=256,
        extra_views=True,
        download=False,
        out=tmp_path / "prepared",
    )
    prepare.prepare(args)
    public = args.out / "public"
    bank = read_json(public / "bank.json")
    key, sha = next(iter(bank["tasks"].items()))
    batch = read_batch(public, key, sha, bank["families"])
    truth = read_json(args.out / "private" / "truth.json")
    task = truth["tasks"][key]
    assert bank["labelfree"]["version"] and len(bank["labelfree"]["sha256"]) == 64
    expected_views = {
        "train",
        "validation",
        "gaussian",
        "refits",
        "projected_real",
        "uniform",
    }
    sampled = np.random.default_rng(5).permutation(len(y))[: args.max_samples]
    fit, select, test = prepare.normal_partitions(y[sampled])
    tr, sel, hidden = prepare.scale(
        X[sampled[fit]], X[sampled[select]], X[sampled[test]]
    )
    np.testing.assert_allclose(batch.views["train"], tr)
    np.testing.assert_allclose(batch.X, sel)
    assert (task["n_fit"], task["n_select"], task["n"]) == (
        len(tr),
        len(sel),
        len(hidden),
    )
    models = [m.fit(tr) for m in prepare.portfolio(5).values()]
    np.testing.assert_allclose(batch.scores, [m.decision_function(sel) for m in models])
    ordered_normals = np.sort(batch.scores, axis=1)
    thresholds = ordered_normals[:, len(sel) - 1 - len(sel) // 20]
    np.testing.assert_array_equal(task["f1_thresholds"], thresholds)
    expected = prepare.oracle(
        y[sampled[test]],
        [m.decision_function(hidden) for m in models],
        f1_thresholds=thresholds,
    )
    for name in expected:
        np.testing.assert_allclose(task["outcomes"][name], expected[name])
    assert set(batch.views) == expected_views
    assert set(np.load(public / f"{key}.npz").files) == {"X", "scores", *batch.views}
    assert "fixture" not in (public / "bank.json").read_text()
    trial = tmp_path / "baselines"
    run(
        Namespace(
            bank=public,
            out=trial,
            selector=None,
            hypothesis=None,
            ireos_rows=None,
            timeout=30,
        )
    )
    submission = read_json(trial / "submission.json")
    outcomes = submission["tasks"][key]
    assert all(r["status"] == "ok" for r in outcomes.values()), outcomes
    assert len(outcomes) >= 28
    assert set(task["outcomes"]) == set(TARGETS)
    out = tmp_path / "grade"
    target = "f1"
    grade(
        Namespace(
            truth=args.out / "private",
            submissions=[trial],
            out=out,
            reference="iforest_default",
            target=target,
        )
    )
    report = read_json(out / "report.json")
    assert report["target"] == target
    values = task["outcomes"][target]
    assert report["supervised_winners"][key] == [
        n for n, v in zip(truth["candidates"], values, strict=True) if v == max(values)
    ]
    assert (
        report["summary"]["iforest_default"][f"{target}_regret"]
        == max(values) - values[0]
    )
    assert report["summary"]["iforest_default"][f"delta_{target}"] == 0
    assert report["summary"]["iforest_default"]["ap"] == task["outcomes"]["ap"][0]
    assert report["by_task"]["iforest_default"][0]["n_fit"] == task["n_fit"]


def test_static_screen_rejects_file_and_process_access(tmp_path, bank):
    from benchmarks.screen import screen

    assert screen("import numpy as np\ndef select(batch):\n    return [1, 2]") == []
    for source in (
        "import os\ndef select(batch): return [1, 2]",
        "from pathlib import Path\ndef select(batch): return [1, 2]",
        "def select(batch): return open('key').read()",
        "import numpy as np\ndef select(batch): return np.load('truth.npz')",
        "def score(batch): return [1, 2]",
    ):
        assert screen(source), source
    public, _ = bank
    plugin = tmp_path / "peek.py"
    plugin.write_text("import os\ndef select(batch): return [1, 2]")
    out = tmp_path / "peek"
    with pytest.raises(ValueError, match="static screen"):
        run(
            Namespace(
                bank=public,
                out=out,
                selector=plugin,
                hypothesis=None,
                ireos_rows=None,
                timeout=30,
            )
        )
    assert not out.exists()


def test_memory_check_links_claims_to_the_ledger(tmp_path):
    from benchmarks.memory import check

    root = tmp_path / "research"
    (root / "claims").mkdir(parents=True)
    (root / "README.md").write_text("# Index\n")
    (root / "ledger.jsonl").write_text(json.dumps({"run": "0123456789ab"}) + "\n")
    claim = root / "claims" / "C-001.md"
    claim.write_text("# C-001: A claim\nstatus: supported\nevidence: 0123456789ab\n")
    assert check(root) == []
    claim.write_text("# C-001: A claim\nstatus: supported\nevidence: ffffffffffff\n")
    assert any("not in the ledger" in p for p in check(root))
    claim.write_text("# C-001: A claim\nstatus: maybe\n")
    assert any("status must be" in p for p in check(root))
    (root / "README.md").write_text("line\n" * 151)
    assert any("limit 150" in p for p in check(root))


def test_held_out_banks_need_the_ci_secret(tmp_path, monkeypatch):
    monkeypatch.delenv(prepare.SALT_VARIABLE, raising=False)
    assert prepare.grading_salt("dev") is None
    assert prepare.task_seed(None, "x", 3) == 3
    for split in ("validation", "test"):
        with pytest.raises(PermissionError, match="held-out CI workflow"):
            prepare.grading_salt(split)
    seeds = {prepare.task_seed(s, d, 0) for s in ("a", "b") for d in ("x", "y")}
    assert len(seeds) == 4 and prepare.task_seed("a", "x", 0) == prepare.task_seed(
        "a", "x", 0
    )


def test_ledger_signatures_and_memory_check(tmp_path):
    from benchmarks.heldout import sign, verify_ledger
    from benchmarks.memory import check

    root = tmp_path / "research"
    root.mkdir()
    ledger = root / "ledger.jsonl"
    row = {"run": "0123456789ab", "split": "validation", "ci_run": "https://ci/1"}
    signed = {**row, "signature": sign(row, "secret")}
    dev = {"run": "ba9876543210", "split": "dev"}
    ledger.write_text("\n".join(json.dumps(r) for r in (dev, signed)) + "\n")
    assert verify_ledger(ledger, "secret") == [] and check(root) == []
    assert verify_ledger(ledger, "other")
    forged = {**signed, "results": {"mine": 0.99}}
    ledger.write_text(json.dumps(forged) + "\n" + json.dumps(row) + "\n")
    assert len(verify_ledger(ledger, "secret")) == 2
    assert any("held-out CI workflow" in p for p in check(root))


def _git(cwd, *command):
    import subprocess

    subprocess.run(["git", *command], cwd=cwd, check=True, capture_output=True)


def test_held_out_look_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("pyod")
    import hashlib
    import shutil

    from benchmarks import heldout

    raw = tmp_path / "raw"
    raw.mkdir()
    rng = np.random.default_rng(3)
    X = rng.normal(size=(240, 4))
    y = (np.arange(len(X)) % 6 == 0).astype(int)
    X[y == 1] += 3
    rows = [["dataset", "source", "split", "git_blob"]]
    for name in ("one", "two"):
        np.savez(raw / f"{name}.npz", X=X, y=y)
        data = (raw / f"{name}.npz").read_bytes()
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        rows += [[name, name, "validation", blob], [name, name, "test", blob]]
    manifest = tmp_path / "datasets.csv"
    with manifest.open("w", newline="") as file:
        csv.writer(file).writerows(rows)
    monkeypatch.setattr(prepare, "MANIFEST", manifest)

    lab = tmp_path / "lab"
    experiment = lab / "research" / "experiments" / "trial"
    experiment.mkdir(parents=True)
    selector = experiment / "selector.py"
    selector.write_text(
        '"""Prefer KNN."""\ndef select(batch):\n'
        '    return [float(f == "KNN") for f in batch.families]\n'
    )
    prediction = experiment / "prediction.md"
    prediction.write_text("research/experiments/trial/selector.py should hold.\n")
    _git(lab, "init", "-q")
    _git(lab, "-c", "user.name=t", "-c", "user.email=t@t", "add", ".")
    _git(lab, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    monkeypatch.chdir(lab)
    rel = Path("research/experiments/trial/selector.py")
    ledger = Path("research/ledger.jsonl")

    def args(split, **extra):
        return Namespace(
            split=split,
            selectors=[str(rel)],
            prediction=Path("research/experiments/trial/prediction.md"),
            target="auroc",
            reference="iforest_default",
            extra_views=False,
            max_samples=2000,
            raw_dir=raw,
            ledger=ledger,
            looks=Path("research/looks"),
            private_out=tmp_path / "private.json",
            ci_run="https://ci/run/1",
            sandbox=__import__("os").environ.get("LABFREE_TEST_SANDBOX"),
            **extra,
        )

    monkeypatch.delenv(prepare.SALT_VARIABLE, raising=False)
    with pytest.raises(PermissionError):
        heldout.look(args("validation"))
    monkeypatch.setenv(prepare.SALT_VARIABLE, "secret")
    heldout.look(args("validation"))
    assert prepare.SALT_VARIABLE not in __import__("os").environ
    (entry,) = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert entry["split"] == "validation" and entry["ci_run"] == "https://ci/run/1"
    assert heldout.verify_ledger(ledger, "secret") == []
    report = read_json(Path(f"research/looks/{entry['run']}-report.json"))
    assert "by_task" not in report and entry["report"].endswith("-report.json")
    (mine,) = [k for k in report["summary"] if k.startswith("selector@")]
    counts = report["summary"][mine]["status_counts"]
    assert counts["ok"] == report["summary"][mine]["tasks"], counts

    monkeypatch.setenv(prepare.SALT_VARIABLE, "secret")
    heldout.look(args("test"))
    test_entry = json.loads(ledger.read_text().splitlines()[-1])
    assert set(test_entry["results"].values()) <= {"promote", "hold"}
    assert "source_wins" in json.dumps(read_json(tmp_path / "private.json"))
    monkeypatch.setenv(prepare.SALT_VARIABLE, "secret")
    with pytest.raises(ValueError, match="one test look"):
        heldout.look(args("test"))

    # A different secret samples different rows, so grades cannot be rebuilt
    # from the public files and seeds alone.
    ledger.write_text("")
    shutil.rmtree("research/looks")
    monkeypatch.setenv(prepare.SALT_VARIABLE, "other")
    heldout.look(args("validation"))
    assert json.loads(ledger.read_text())["bank"] != entry["bank"]

    prediction.write_text("Names nothing.\n")
    _git(lab, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "y")
    monkeypatch.setenv(prepare.SALT_VARIABLE, "other")
    with pytest.raises(ValueError, match="name every selector"):
        heldout.look(args("validation"))
    outside = tmp_path / "elsewhere.py"
    outside.write_text(selector.read_text())
    with pytest.raises(ValueError, match="under research/experiments"):
        heldout.check_request(
            "validation", [outside], prediction, "iforest_default", ledger
        )
