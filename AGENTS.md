# Research work

For hypothesis generation, benchmark interpretation or a literature review, start with [the research memory](research/README.md). It links open questions, failed ideas, primary sources and the experiment protocol. Record the evidence and decision there when completing a research iteration.

The primary question is how closely a metric using only known-normal observations can reproduce the detector choice a labeled evaluation would make for a specified score, such as ROC AUC, PR AUC or F1. Fit candidates on normal data and select using normal data. After freezing the choice, compare its score on a hidden mixture of normal observations and real outliers with the best score in the same fitted portfolio. The first cohort studied a historical mixed-data setup. Record the track and supervised target with every result.

## Repository boundaries

- **This repository is the lab.** It holds the benchmark harness (`benchmarks/`), the research memory (`research/`) and nothing else. Run every command from the repository root after `python -m pip install -r requirements.txt`.
- **The metrics live in [`labelfree`](https://github.com/OliverHennhoefer/label-free).** The lab uses the exact library commit pinned in `requirements.txt`. Do not copy or patch library code here. If an experiment exposes a metric bug, record it in the research memory, fix it in `label-free`, then move the pin here. A pin change can change baseline results, so name the library commit with every result.
- **Keep grading data away from candidate code.** Raw ADBench files, `private/` grading keys and run caches stay under the ignored `.bench-runs/` directory or in evaluator-only storage. Never commit them. A selector under study receives only a public bank.
- **Validation and test are reserved.** Use development reports freely. Record every validation or test look in the research memory before making it.
