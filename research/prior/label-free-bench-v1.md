# Prior work: label-free-bench v1 (July 2026)

Before this lab, the repository held a separate normal-only benchmark, `label-free-bench` v1. Its code was removed when the lab moved here, because two harnesses would leave it unclear which protocol is authoritative. This note keeps what v1 established and the design ideas worth testing again. The last v1 revision is commit `0841a31` in this repository's history.

## What v1 measured

The question was the same as the lab's: rank a frozen pool of detector configurations using only normal data, then grade the choice with hidden labels.

- **Data.** 15 classical ADBench datasets with at least 1,400 rows and 50 anomalies. Development: annthyroid, landsat, letter, mammography, optdigits, shuttle, vowels. Confirmation: cardio, fault, musk, PageBlocks, pendigits, satellite, satimage-2, speech.
- **Split.** Each dataset was stratified 50:50. Development anomalies were discarded. Development normals were capped at 500 and were the only rows metrics saw. The labeled half was the test set.
- **Pool.** 60 candidates, 12 per family, for Isolation Forest, LOF, PCA, One-Class SVM and HBOS. The first candidate of each family was the PyOD default. The others came from a deterministic stratified design.
- **Endpoint.** Normalized AP regret: (oracle AP − selected AP) / (oracle AP − worst AP), reported as selection quality = 1 − normalized regret. ROC AUC was secondary.
- **Inference.** Datasets weighted equally, a hierarchical bootstrap over datasets then seeds, ten seeds. Confirmation alone received an exact sign-flip test with Holm correction.
- **Failures.** A failed metric received normalized regret 1, the worst outcome, and stayed in every comparison.

## Reference results

Selection quality with 95% intervals. Higher is better. Ten seeds per dataset.

| Metric | Development (7 datasets) | Confirmation (8 datasets) |
| --- | ---: | ---: |
| `excess_mass` | **0.711** [0.571, 0.838] | 0.584 [0.347, 0.821] |
| `mass_volume` | 0.702 [0.568, 0.823] | 0.554 [0.333, 0.773] |
| `score_calinski_harabasz` | 0.591 [0.435, 0.712] | 0.577 [0.378, 0.773] |
| `default_lof` | 0.549 [0.339, 0.751] | 0.605 [0.391, 0.817] |
| `score_silhouette` | 0.500 [0.332, 0.651] | **0.615** [0.407, 0.813] |
| `npd` | 0.496 [0.280, 0.713] | 0.592 [0.366, 0.819] |
| `eag` | 0.465 [0.323, 0.596] | 0.588 [0.402, 0.776] |
| `sireos` | 0.464 [0.294, 0.640] | 0.556 [0.349, 0.766] |
| `model_centrality` | 0.413 [0.216, 0.632] | 0.535 [0.387, 0.678] |
| `random_candidate` | 0.396 [0.298, 0.491] | 0.528 [0.368, 0.684] |
| `default_iforest` | 0.386 [0.191, 0.606] | 0.490 [0.354, 0.624] |
| `default_ocsvm` | 0.307 [0.142, 0.492] | 0.610 [0.397, 0.822] |
| `ireos` (58.6% / 86.2% success) | 0.317 [0.091, 0.570] | 0.467 [0.231, 0.702] |
| `asoi` | 0.247 [0.070, 0.467] | 0.401 [0.187, 0.631] |
| `laplacian` | 0.275 [0.067, 0.524] | 0.307 [0.096, 0.552] |
| `ranking_stability` | 0.218 [0.041, 0.430] | 0.446 [0.173, 0.709] |

The full v1 table had 24 rows. The rest fell within these ranges.

**What it suggests.**
- EM and MV led development by a wide margin but did not hold that lead on confirmation. There, no metric was clearly better than a fixed default LOF or One-Class SVM.
- Every interval is wide, and on confirmation most of them include random selection.
- These are aggregate results under a different split, pool and endpoint. They are not comparable number for number with the lab's banks.

## Data exposure

v1 published aggregate graded results for datasets that the lab's manifest reserves:

- **Validation:** vowels, musk.
- **Test:** landsat, satellite and satimage-2 (one source group), shuttle, fault, PageBlocks.

No per-dataset outcomes or per-task picks were published. A method designed with the v1 table in view has still seen aggregate feedback that includes these sources. Disclose this in any paper, and consider an external replication set that v1 never touched.

## Ideas to reconsider

Each idea changes the frozen protocol. Adopting one means a new bank and a recorded decision, not a silent edit.

| v1 idea | Why it may matter | Lab status |
| --- | --- | --- |
| Larger pool with PCA and One-Class SVM, 12 configurations per family | Default OCSVM and LOF were among the strongest confirmation choices. The lab pool has 8 candidates in 4 families, so it cannot test within-family tuning or these families. | Open. The benchmark README already asks for a wider frozen pool before broad claims. |
| Normalized regret | Makes regret comparable across datasets with very different oracle ranges | Open. The lab reports raw regret and best-pick rate. |
| AP at 1%, 5% and 10% prevalence by class reweighting | Tests whether a choice survives a shift in anomaly prevalence without resampling | Open |
| Failure counts as worst case | Stops a method from looking good by abstaining where it would be wrong. The lab falls back to default Isolation Forest and reports status counts. | Open. Report both conventions if they differ. |
| Exact sign-flip test with Holm correction, one confirmatory run only | A preregistered final comparison | Open. The lab's test split is reserved for exactly one frozen look. |
| Nested candidate budgets | Shows whether a method's advantage depends on pool size | Open |
