"""Grade frozen rankings and append every look to the run ledger.

Disclosure depends on the split. Development reports are complete. Validation
reports give aggregates rounded to two decimals. Test reports give only a
promote-or-hold decision per selector from a one-sided sign test over source
groups. Validation and test looks require a committed prediction file.
"""

import json
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr

from .protocol import (
    DEFAULT,
    TARGETS,
    VERSION,
    digest,
    new_directory,
    read_json,
    write_json,
)

DEFAULT_POLICY = "iforest_default"
RANDOM_POLICY = "uniform_model"
ALPHA = 0.05


def committed(path: Path) -> str:
    """Return the commit that holds this file unchanged, or raise."""
    path = path.resolve()

    def git(*command):
        return subprocess.run(
            ["git", *command],
            cwd=path.parent,
            capture_output=True,
            text=True,
            check=False,
        )

    tracked = git("ls-files", "--error-unmatch", path.name).returncode == 0
    clean = git("diff", "--quiet", "HEAD", "--", path.name).returncode == 0
    if not (path.is_file() and tracked and clean):
        raise ValueError(
            f"Commit the prediction before a held-out look: {path}. "
            "It must be tracked and unchanged since the last commit."
        )
    return git("log", "-1", "--format=%H", "--", path.name).stdout.strip()


def repository_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or None


def decisions(records, target):
    """One-sided exact sign test of source-level gains over the reference."""
    gains = group_means(records, f"delta_{target}")
    wins, losses = int(np.sum(gains > 0)), int(np.sum(gains < 0))
    p = binomtest(wins, wins + losses, alternative="greater").pvalue if wins else 1.0
    return {
        "decision": "promote" if p < ALPHA else "hold",
        "source_wins": wins,
        "source_losses": losses,
        "sources": len(gains),
        "p_value": float(p),
    }


def rounded(value):
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [rounded(v) for v in value]
    return value


def disclose(summary, split):
    return rounded(summary) if split == "validation" else summary


def load_trials(paths, truth):
    tasks = {key: {} for key in truth["tasks"]}
    provenance = []
    seen = {DEFAULT_POLICY, RANDOM_POLICY}
    seen.update(f"fixed/{name}" for name in truth["candidates"])
    for directory in paths:
        path = directory / "submission.json"
        trial = read_json(path)
        if trial["version"] != VERSION or trial["bank_id"] != truth["bank_id"]:
            raise ValueError(
                f"Submission belongs to a different benchmark bank: {path}"
            )
        if trial["tasks"].keys() != tasks.keys():
            raise ValueError(f"Incomplete or unexpected set of tasks in {path}")
        names = set(next(iter(trial["tasks"].values())))
        if not names or seen & names:
            raise ValueError(
                f"Empty or duplicate selector names in {path}: {seen & names}"
            )
        for key, results in trial["tasks"].items():
            if set(results) != names:
                raise ValueError(f"Missing selector results on {key}")
            for item in results.values():
                if item["status"] not in {"ok", "abstained", "failed", "unavailable"}:
                    raise ValueError("Unknown submission status")
                if item["status"] == "ok":
                    ranks = np.asarray(item["ranks"], dtype=float)
                    if ranks.shape != (len(truth["candidates"]),) or not np.array_equal(
                        ranks, rankdata(ranks)
                    ):
                        raise ValueError("Invalid candidate ranks")
            tasks[key].update(results)
        seen.update(names)
        provenance.append(
            {
                "sha256": digest(path),
                "methods": sorted(names),
                "hypothesis": trial["hypothesis"],
                **trial["provenance"],
                "selector_sha256": trial.get("selector_sha256"),
            }
        )
    return tasks, provenance


def evaluate_choice(item, outcomes, candidates, default):
    status = item["status"]
    ranks = np.asarray(item["ranks"]) if status == "ok" else None
    chosen = np.flatnonzero(ranks == ranks.min()) if ranks is not None else [default]
    quality = {
        name: float(np.mean(np.asarray(values)[chosen]))
        for name, values in outcomes.items()
    }
    diagnostics = {}
    for name, values in outcomes.items():
        best = max(values)
        diagnostics[f"{name}_regret"] = best - quality[name]
        diagnostics[f"{name}_best_pick_rate"] = float(
            np.mean(np.asarray(values)[chosen] == best)
        )
        correlation = None
        if ranks is not None and np.ptp(ranks) and np.ptp(values):
            correlation = float(spearmanr(-ranks, values).statistic)
        diagnostics[f"spearman_{name}"] = correlation
    return {
        **quality,
        "status": status,
        "selected": [candidates[i] for i in chosen],
        "reason": item.get("reason"),
        "seconds": item.get("seconds"),
        **diagnostics,
    }


def group_means(records, column):
    # First collapse repeat seeds, then treat variants of one source as a block.
    groups = defaultdict(lambda: defaultdict(list))
    for row in records:
        if row[column] is not None:
            groups[row["source"]][row["dataset"]].append(row[column])
    return np.array(
        [
            np.mean([np.mean(v) for v in datasets.values()])
            for _, datasets in sorted(groups.items())
        ]
    )


def summarize(records, target="ap"):
    endpoints = [e for e in TARGETS if e in records[0]]
    columns = (
        *endpoints,
        *(f"{e}_regret" for e in endpoints),
        *(f"{e}_best_pick_rate" for e in endpoints),
        *(f"spearman_{e}" for e in endpoints),
        *(f"delta_{e}" for e in endpoints),
    )
    means = {c: group_means(records, c) for c in columns}
    delta = means[f"delta_{target}"]
    intervals = {f"delta_{e}_ci95": None for e in endpoints}
    if len(delta) >= 2:
        draws = np.random.default_rng(0).integers(len(delta), size=(5000, len(delta)))
        for e in endpoints:
            replicates = means[f"delta_{e}"][draws].mean(axis=1)
            intervals[f"delta_{e}_ci95"] = np.quantile(
                replicates, [0.025, 0.975]
            ).tolist()
    statuses = {
        s: sum(row["status"] == s for row in records)
        for s in ("ok", "abstained", "failed", "unavailable")
    }
    return {
        **{c: float(v.mean()) if len(v) else None for c, v in means.items()},
        **intervals,
        "sources": len(delta),
        "source_wins": int(np.sum(delta > 0)),
        "tasks": len(records),
        "status_counts": statuses,
    }


def grade(args):
    truth = read_json(args.truth / "truth.json")
    target = getattr(args, "target", "ap")
    split = truth["split"]
    prediction = getattr(args, "prediction", None)
    prediction_commit = None
    if split != "dev":
        if prediction is None:
            raise ValueError(f"A {split} look needs --prediction <committed file>")
        prediction_commit = committed(prediction)
    submissions, provenance = load_trials(args.submissions, truth)
    candidates = truth["candidates"]
    default = candidates.index(DEFAULT)
    controls = {RANDOM_POLICY: {"status": "ok", "ranks": [1] * len(candidates)}}
    for index, name in enumerate(candidates):
        policy = DEFAULT_POLICY if index == default else f"fixed/{name}"
        controls[policy] = {
            "status": "ok",
            "ranks": rankdata(np.arange(len(candidates)) != index).tolist(),
        }
    records, skipped, supervised_winners = defaultdict(list), [], {}
    for key, task in truth["tasks"].items():
        outcomes = task["outcomes"]
        if outcomes is None:
            skipped.append(key)
            continue
        if target not in outcomes:
            raise ValueError(f"Target {target!r} is unavailable in this grading key.")
        best = {e: max(scores) for e, scores in outcomes.items()}
        if truth["split"] == "dev":
            supervised_winners[key] = [
                name
                for name, score in zip(candidates, outcomes[target], strict=True)
                if score == best[target]
            ]
        choices = {**submissions[key], **controls}
        assessed = {
            name: evaluate_choice(item, outcomes, candidates, default)
            for name, item in choices.items()
        }
        if args.reference not in assessed:
            raise ValueError(
                f"Reference selector missing from submissions: {args.reference}"
            )
        reference = assessed[args.reference]
        for name, values in assessed.items():
            records[name].append(
                {
                    **values,
                    "task": key,
                    "dataset": task["dataset"],
                    "source": task["source"],
                    "seed": task["seed"],
                    "n": task["n"],
                    **{k: task[k] for k in ("n_fit", "n_select") if k in task},
                    "features": task["features"],
                    **{f"oracle_{e}": value for e, value in best.items()},
                    **{f"delta_{e}": values[e] - reference[e] for e in outcomes},
                }
            )
    if not records:
        raise ValueError("No batch has both classes. Increase the label-blind row cap.")
    report = {
        "version": VERSION,
        "bank_id": truth["bank_id"],
        "split": split,
        "target": target,
        "reference": args.reference,
        "submissions": provenance,
        "grader_sha256": digest(Path(__file__)),
        "single_class_tasks": len(skipped),
    }
    if split == "test":
        # Win counts and p-values would rank finalists that all hold, so they
        # stay with the grading key. Only the decision leaves the grader.
        private_tests = {
            name: decisions(rows, target)
            for name, rows in sorted(records.items())
            if name != args.reference
        }
        report["decisions"] = {
            name: row["decision"] for name, row in private_tests.items()
        }
    else:
        report["summary"] = disclose(
            {name: summarize(rows, target) for name, rows in sorted(records.items())},
            split,
        )
    if split == "dev":
        report["by_task"] = dict(records)
        report["supervised_winners"] = supervised_winners
        report["single_class_task_ids"] = skipped
    with new_directory(args.out) as output:
        write_json(output / "report.json", report)
        report_sha = digest(output / "report.json")
    if split == "test":
        looks = args.truth / "looks"
        looks.mkdir(mode=0o700, exist_ok=True)
        write_json(looks / f"{report_sha[:12]}.json", private_tests)
    append_ledger(
        getattr(args, "ledger", None),
        report,
        report_sha,
        args.out,
        prediction,
        prediction_commit,
        signer=getattr(args, "signer", None),
        ci_run=getattr(args, "ci_run", None),
        published=getattr(args, "published", None),
    )
    if split == "test":
        print(f"Test decisions against {args.reference} on target {target}")
        for name, decision in report["decisions"].items():
            print(f"{name:36} {decision:>8}")
        return
    print(f"Results for supervised target {target}")
    print(
        f"{'selector':36} {'score':>7} {'gap':>8} {'vs ref':>9} {'best pick':>10} {'scored':>9}"
    )
    for name, row in report["summary"].items():
        print(
            f"{name:36} {row[target]:7.3f} {row[f'{target}_regret']:8.3f} "
            f"{row[f'delta_{target}']:+9.3f} {row[f'{target}_best_pick_rate']:10.1%} "
            f"{row['status_counts']['ok']:5}/{row['tasks']}"
            + (" *" if row["status_counts"]["ok"] < row["tasks"] else "")
        )
    if any(r["status_counts"]["ok"] < r["tasks"] for r in report["summary"].values()):
        print(
            "* Did not rank every task. Unranked tasks use default Isolation Forest's choice."
        )
    if skipped:
        print(f"{len(skipped)} single-class batches omitted for all selectors.")


def append_ledger(
    ledger,
    report,
    report_sha,
    out,
    prediction,
    prediction_commit,
    *,
    signer=None,
    ci_run=None,
    published=None,
):
    """Append one look. The ledger is the only source for numbers in research notes.

    Held-out looks come from CI, which records its run URL and signs the row with
    the grading secret so that a hand-written or edited row fails verification.
    """
    if ledger is None:
        return
    target = report["target"]
    if "decisions" in report:
        results = report["decisions"]
    else:
        results = {
            name: {
                target: row[target],
                f"{target}_regret": row[f"{target}_regret"],
                f"delta_{target}": row[f"delta_{target}"],
                "ok": row["status_counts"]["ok"],
                "tasks": row["tasks"],
            }
            for name, row in report["summary"].items()
        }
    entry = {
        "run": report_sha[:12],
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "split": report["split"],
        "bank": report["bank_id"][:12],
        "target": target,
        "reference": report["reference"],
        "repo_commit": repository_commit(),
        "prediction": str(prediction) if prediction else None,
        "prediction_commit": prediction_commit,
        "report": str(published / f"{report_sha[:12]}-report.json")
        if published
        else str(out),
        "selectors": {
            name: sha[:12]
            for submission in report["submissions"]
            for name in submission["methods"]
            if (sha := submission.get("selector_sha256"))
        },
        "results": results,
    }
    if report["split"] != "dev":
        entry["ci_run"] = ci_run
    if signer is not None:
        entry["signature"] = signer(entry)
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a") as stream:
        stream.write(json.dumps(entry, allow_nan=False) + "\n")
    print(f"Ledger entry {entry['run']} appended to {ledger}")
