# Batch model-selection benchmark

The experiment asks how closely a metric using only **known-normal examples** can reproduce the detector choice a supervised evaluation would make. Name a target such as ROC AUC, PR AUC or F1 before grading. Compare the hidden test score of the chosen detector with the best test score among the same fitted candidates. This difference is the **selection gap** (regret). If the best ROC AUC is 0.94 and the chosen detector achieves 0.93, the gap is 0.01. The metric used to choose can have any numerical scale. The metrics come from [`labelfree`](https://github.com/OliverHennhoefer/label-free), pinned to an exact commit in `requirements.txt`.

## Normal-only protocol

1. A trusted preparer shuffles and samples up to `--max-samples` original rows without replacement (default 2,000). It reserves one quarter of each available class, rounded up, for the private test. This approximately preserves the sampled class mix. The other outliers are unused.
2. Of the remaining normal rows, two thirds (rounded down) fit the imputer, feature mask, scaler and detectors. The rest are separate normal examples for choosing a detector. These groups and the private group have disjoint row indices. The public bank contains the fitting normals, the selection normals and the fitted detectors' scores on the selection normals.
3. The selector freezes its ranking from that public bank. The private grader evaluates the same fitted detectors on the reserved mix of normal and outlier rows. The trusted preparer may cache these outcomes ahead of time. The researcher receives the permitted grade after submitting the choice. Public data contains neither private rows nor their scores or labels. A private sample with only one class is ungradable.

The basic bank contains only real, normal observations. `--extra-views` is an explicit comparison condition for baselines that use normal-data refits and artificial Gaussian or uniform reference points. All those views are constructed solely from the fitting and selection normals. Record this option when interpreting a method that assumes a type of artificial anomaly.

For agent-led research, start with the [research memory](../research/README.md).

## Try a development iteration

Run these commands from the repository root with Python 3.12. The first download uses the official ADBench repository. Subsequent runs reuse local files.

```sh
python -m pip install -r requirements.txt

run=.bench-runs/normal-only
python -m benchmarks prepare --download --raw-dir .bench-runs/raw \
  --split dev --datasets 4_breastw 6_cardio 18_Ionosphere 29_Pima \
  --seeds 0 1 --out "$run/dev" --key-dir ../lab-keys/normal-only-dev

python -m benchmarks run --bank "$run/dev/public" \
  --ireos-rows 24 --out "$run/baselines"

python -m benchmarks run --bank "$run/dev/public" \
  --selector benchmarks/example_selector.py \
  --out "$run/trial-001"

python -m benchmarks grade --truth ../lab-keys/normal-only-dev \
  --submissions "$run/baselines" "$run/trial-001" \
  --target auroc --reference iforest_default --out "$run/report-001"
```

For the next iteration, run the new selector against the **same public directory**. Grade both trial directories together with the cached baselines. Pass the previous selector's full name from its report as `--reference` for a paired comparison. Each run has a new output path and snapshots its source and hypothesis. Changing the portfolio, evidence, preprocessing or row sample creates a new bank. The grader checks bank IDs and requires every submitted selector to account for every task. Every grade appends one row to `research/ledger.jsonl`, the only source for numbers in research notes. `--key-dir` moves the grading key out of the workspace. Without it the key stays in `<out>/private`.

## Writing a hypothesis

A standalone Python module exports `select(batch)`. Before it runs, a static screen rejects imports and calls that read files, reach the network, start processes or execute generated code. The screen is a tripwire for honest mistakes and shortcuts, not a sandbox. Return one value per candidate, with larger values preferred, or return `None` to abstain. The module docstring records the hypothesis. The CLI also accepts `--hypothesis`. A fresh Python process evaluates each custom selector on each task, with a default wall-clock budget of 120 seconds including startup. Set `--timeout` to change it. Each task gets read-only arrays.

| Input | Meaning |
| --- | --- |
| `batch.X` | Normal selection rows transformed using the fitting normals |
| `batch.scores` | Candidate × selection row. Larger values mean more anomalous |
| `batch.families` | One detector family per candidate, in matrix order |
| `batch.views` | `train` contains the transformed fitting normals. Optional additional views are described below. |

The [example selector](example_selector.py) reproduces the library's model-centrality baseline to show the contract. It is an interface example, not a hypothesis. Development reports include per-task picks, oracle scores, sample and feature counts, and error reasons. These let an agent inspect a failed hypothesis without refitting the portfolio.

## Frozen protocol

- **Data.** The manifest lists all 47 ADBench Classical files and their Git blob hashes at one revision. It assigns 25 files to development, eight to validation and 14 to test. Related variants stay together, including the Cardiotocography, thyroid, Wisconsin breast and Landsat groups. The six datasets already used by the `labelfree` example notebooks are in development. Treat these source groups as an initial provenance audit. [Manifest](adbench.csv) · [Pinned ADBench collection](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/datasets/Classical)
- **Unit of evaluation.** One dataset and seed define one task, split three ways as above. The fixed 5% alert budget ranks even a completely normal selection set. It does not estimate the prevalence of anomalies in that set.
- **Candidate pool.** Two configurations each for Isolation Forest, KNN, LOF and HBOS. Isolation Forest uses its default sample count or 64, KNN and LOF use 5 or 25 neighbors, and HBOS uses 5 or 10 bins. Stochastic fits use the task seed. This small, family-balanced pool is intended for development. Freeze a wider pool and test sensitivity to duplicating or removing detector families before making a broad selection claim.
- **Cheap and expensive signals.** The basic bank supports all ten score-cluster indices, ASI/ASOI, RTM/EAG, Laplacian Score, SIREOS, MC/MCS/KNN centrality, UDR, HITS, ENS, and Borda and robust Borda over ASOI, Laplacian, SIREOS, MC and HITS. The `labelfree` notebooks aggregate the same five metrics but run Laplacian with 10 neighbors. The benchmark uses the library default of 5. The fixed alert prior is 5% for partition and stability metrics. The adapters preserve each metric's native direction and apply score unification where needed. `--ireos-rows N` adds an explicitly named, shared-gamma IREOS comparison on the first N shuffled rows, capped at the batch size. It grades the resulting choice on the private target. The 24-row example is a smoke test.
- **Extra views.** NPD compares selection scores with scores on diagonal-Gaussian probes. Stability uses five 80% subsample refits scoring the same selection rows. EM/MV use 2,048 uniform reference points per projected unit box, with up to three five-feature projections. The box is defined by the available fitting and selection features. Arrays are `validation` and `gaussian` (candidate × row), `refits` (candidate × refit × row), and `projected_real` and `uniform` (projection × candidate × row). These methods report unavailable on a basic bank. All real rows in these views are normal, and the main fitted detectors score the hidden test once for grading.
- **Disclosure by split.** Development reports are complete, including per-task picks and supervised winners. Validation reports give aggregates rounded to two decimals. Test reports give only a promote-or-hold decision per selector against the reference: a one-sided exact sign test over source groups at α = 0.05. The 14 test files form 10 source groups, so promotion needs at least 9 sources to improve when none ties. Reserve test for one or two finalists. Validation and test looks require `--prediction`, a file committed to git and unchanged since, whose commit the ledger records.
- **Reproducibility.** Inputs are checked against their pinned hashes. The public manifest records the environment versions, the `labelfree` version, commit and source hash, seeds, protocol and task-file hashes. Submissions contain source fingerprints, a custom selector's source snapshot, per-task ranks, status and timing. Reports include submission and grader fingerprints. Store the bank, inputs, environment and submissions used for a publication.

## Held-out looks in CI

Development runs locally and costs nothing. Validation and test looks run only in the [held-out workflow](../.github/workflows/heldout.yml), so the grading key never exists where agents work.

**Why a secret.** ADBench labels are public, and the manifest says which datasets are held out. With public seeds anyone could rebuild the held-out rows and labels from the pinned files. The workflow keys each task's row sample and detector seeds on `LABFREE_GRADING_SALT`, an HMAC secret held in the repository's `grading` environment. The dataset is known, but the graded rows are not. Held-out banks always use seeds 0, 1 and 2, so a look cannot be repeated with fresh seeds.

**One look.** Someone dispatches the workflow on `main` with the split, the selector paths, the committed prediction, the target, the reference and whether to build extra views. The job:

1. Refuses the request unless every selector is a committed `.py` under `research/experiments/`, the prediction is committed and names each selector path, and, for test, there are at most two finalists that have never had a test look. It also verifies the signature of every held-out ledger row.
2. Builds the held-out bank with the secret, then removes the secret from its environment and the grading key from disk.
3. Runs each selector as an unprivileged `candidate` user in a network namespace without network. That user cannot read the grader's environment or the key, and cannot write the checkout or the Python installation.
4. Grades, signs the ledger row with the secret, and records the CI run URL. It pushes the row and the disclosed report (`research/looks/<run>-report.json`) to `main`. For test it also commits the private win counts, encrypted with the secret, as `research/looks/<run>-private.json.enc`.

The job prints only what the split discloses, so its public logs reveal no more than the committed report. `check-memory` rejects a held-out ledger row without a CI run and signature, and every look re-verifies all signatures, so a hand-written or edited row is caught.

**Request a look** from the Actions tab, or with the GitHub CLI:

```sh
gh workflow run heldout.yml -f split=validation \
  -f selectors="research/experiments/<slug>/selector.py" \
  -f prediction=research/experiments/<slug>/prediction.md \
  -f target=ap -f reference=iforest_default
git pull  # the ledger row arrives on main
```

**Cost.** The workflow runs only on an explicit dispatch, with no matrix and no schedule. It caches pip and the public raw files, never the bank or the key. Standard runners are free for public repositories. A look takes a few minutes, mostly installing dependencies and fitting the 8-candidate pool.

**One-time setup** (repository admin):

1. Create the environment **Settings → Environments → New environment** named `grading`. Under deployment branches choose *Selected branches* and add `main` only. Optionally add yourself as a required reviewer, so every look needs your click.
2. Add the environment secret `LABFREE_GRADING_SALT`, for example the output of `python -c "import secrets; print(secrets.token_hex(32))"`. Keep a copy offline to decrypt private test files: `openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 -pass env:LABFREE_GRADING_SALT -in <file>.enc`. Changing it creates new held-out banks.
3. Protect `main` so that changes, including to this workflow, arrive by reviewed pull request. If the protection blocks direct pushes, allow GitHub Actions to bypass it or the ledger push fails.
4. Under **Settings → Actions → General**, keep *Require approval for fork pull request workflows* on. Forks never receive environment secrets, and the workflow has no pull-request trigger.

**What remains open.** The held-out dataset names are public. A selector could hard-code choices for recognizable datasets. The screen does not catch that, so a verifier reads the selector code before trusting a held-out result. A provenance-disjoint replication remains the standard for a paper. The same command runs outside GitHub: `python -m benchmarks heldout ... --sandbox <prefix>` on any machine that holds the secret and runs candidates under another account, for example a separate user on a home server. GitHub is the default because the key then never touches a machine that agents use.

## Reading a report

In a development report, `n_fit` and `n_select` count the normal rows for fitting and selection, and `n` counts the private test rows. Set `--target` to `auroc`, `pr_auc_trapezoid`, `ap`, `f1` or `precision_at_budget`. `ap` remains the command's default. `pr_auc_trapezoid` integrates the precision–recall curve by trapezoids. `ap` uses non-interpolated average precision, which ADBench calls `aucpr`. `precision_at_budget` evaluates the top 5% of test scores. [ADBench grader](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/myutils.py) · [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

**F1 needs an alert cutoff.** For each detector the preparer takes the empirical 95th percentile of its scores on the selection normals (inverse empirical CDF). A test score strictly above that fixed cutoff is an alert. This flags at most 5% of the selection normals, with ties at the cutoff treated as normal. The actual alert rate on new data can differ. The grader computes ordinary binary F1 from the test labels at that cutoff.

For each available supervised score the report gives the chosen detector's score and gap to the supervised best. `*_best_pick_rate` is the probability of picking one of the supervised winners under the tie policy. Within-task Spearman correlation measures agreement over the whole ranking where defined. A development report also names the actual `supervised_winners` on each task for the declared target. Validation and test expose aggregates only.

Tied winners receive the expected utility of a uniform choice among them. `uniform_model` is the mean utility of every candidate. Fixed controls expose the default Isolation Forest and every other individual configuration. Choose a strong fixed control on development and freeze it before validation. A failed, unavailable or abstaining selector falls back to default Isolation Forest. Status counts expose exactly how often it supplied a ranking. A single-class batch is ungradable and is omitted for every method, with its count reported.

The report averages seeds within datasets, datasets within source groups and source groups equally. Intervals for paired differences to the reference resample whole source groups 5,000 times. `source_wins` counts sources where the selector beats the reference on the declared target. A one-source run has no interval. Four-source smoke results can test the software and suggest a failure mode. A publication claim needs more independent sources and a frozen comparison.

## Giving agents access

The example commands run all roles locally for development. Held-out looks follow [Held-out looks in CI](#held-out-looks-in-ci). For a blind evaluation outside this repository, a trusted preparer keeps the raw NPZs and the grading key in grader-only storage, for example under another user account. Give the candidate execution environment a read-only copy of `public`, its method code, a writable submission destination and no network access. An external evaluator can call `grade` and return a report without executing candidate code. The surrounding execution environment supplies filesystem and network isolation.

Let agents query development reports freely. Freeze a shortlist before returning validation feedback and run the final test once on a frozen revision. The ledger records every look. Repeated aggregate scores can still be used to overfit a held-out split, which is why validation is rounded and test gives decisions only. Public ADBench datasets can also be recognized. Reserve a provenance-disjoint external replication for a paper. [Blum and Hardt on adaptive leaderboards](https://proceedings.mlr.press/v37/blum15.html) · [MLE-bench's submission and grading design](https://github.com/openai/mle-bench)

An agent or researcher owns the hypothesis, decision rule, interpretation and next experiment. The benchmark's regression tests run with `python -m pytest`, including the PyOD integration test. The metric formulas are tested in the `labelfree` repository.
