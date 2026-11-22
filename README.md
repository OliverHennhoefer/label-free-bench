# Label-Free Laboratory

A research lab for one question: **if you only have examples of normal behavior, how closely can you choose the anomaly detector that a labeled evaluation would choose?** Candidate PyOD detectors are fitted on known-normal ADBench records. A label-free selector ranks them using other normal records and freezes its choice. A separate grader then scores every candidate on a hidden mix of normal records and real outliers. It reports how much ROC AUC, average precision, PR AUC or F1 the choice gave up against the best candidate.

The lab is built for iterative research by people and agents. Hypotheses are written as small selector modules, predictions are recorded before grading, and each result, including failures, goes into a searchable research memory.

| Where | What |
| --- | --- |
| [`AGENTS.md`](AGENTS.md) | Rules for agents: repository boundaries, data handling, reserved splits |
| [`research/`](research/README.md) | Research memory: open questions, experiments, literature reviews, prior work |
| [`benchmarks/`](benchmarks/README.md) | The benchmark: protocol, commands, selector interface, report fields |

The metrics come from [`labelfree`](https://github.com/OliverHennhoefer/label-free), pinned to an exact commit in [`requirements.txt`](requirements.txt).

## Quick start

Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m pytest
```

Then follow [Try a development iteration](benchmarks/README.md#try-a-development-iteration) to prepare a score bank, run the library baselines and grade a selector.

## License

MIT. ADBench datasets are downloaded from their pinned source and are not redistributed. Cite ADBench and follow the terms of the upstream datasets.
