# Dataset registry

V1 uses 15 canonical classical ADBench datasets. They were selected before metric
comparison to provide at least 1,400 observations, at least 50 anomalies, varied
dimensionality, varied source prevalence, and broad domains while remaining
practical on an ordinary CPU. Files, SHA-256 hashes, categories, and suite
assignments are frozen in `src/label_free_bench/datasets.json`.

| Suite | Datasets |
|---|---|
| Development | annthyroid, landsat, letter, mammography, optdigits, shuttle, vowels |
| Confirmation | cardio, fault, musk, PageBlocks, pendigits, satellite, satimage-2, speech |

The split is fixed before metric discovery. It balances the available domains as
far as this small collection permits; results include per-domain and leave-one-
domain-out summaries because datasets from the same broad domain are not assumed
to establish independent domain generalization.

Download and verify the exact files with:

```bash
uv run label-free-bench fetch-data
```

ADBench aggregates datasets originating from multiple upstream sources. Users
must cite ADBench and follow any upstream dataset terms applicable to their use.
The data are not redistributed by this repository.
