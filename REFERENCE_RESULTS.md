# V1 reference results

Source-prevalence results for schema v5 and candidate-pool v1.1 are below.
Selection quality is `1 - normalized_average_precision_regret`; higher is better.
Intervals are deterministic 95% hierarchical-bootstrap intervals over datasets
and seeds. Full four-prevalence bundles with manifests, frozen normal-only
evidence, and audit tables live in `results/reference-{development,confirmation}`
and should be attached to the v1.0.0 release rather than committed (~84 MB).

## Development

Seven datasets, ten seeds, 4,200 candidates; exploratory inference.

| Rank | Metric | Selection quality (95% CI) | Mean selected AP (95% CI) | Success |
|---:|---|---:|---:|---:|
| 1 | `excess_mass` | 0.711 [0.571, 0.838] | 0.475 [0.295, 0.670] | 100.0% |
| 2 | `mass_volume` | 0.702 [0.568, 0.823] | 0.472 [0.298, 0.669] | 100.0% |
| 3 | `score_calinski_harabasz` | 0.591 [0.435, 0.712] | 0.429 [0.259, 0.617] | 100.0% |
| 4 | `default_lof` | 0.549 [0.339, 0.751] | 0.432 [0.257, 0.619] | 100.0% |
| 5 | `score_davies_bouldin` | 0.535 [0.384, 0.679] | 0.393 [0.239, 0.598] | 100.0% |
| 6 | `score_silhouette` | 0.500 [0.332, 0.651] | 0.410 [0.248, 0.594] | 100.0% |
| 7 | `npd` | 0.496 [0.280, 0.713] | 0.411 [0.216, 0.621] | 100.0% |
| 8 | `eag` | 0.465 [0.323, 0.596] | 0.396 [0.219, 0.594] | 100.0% |
| 9 | `sireos` | 0.464 [0.294, 0.640] | 0.399 [0.223, 0.596] | 100.0% |
| 10 | `average_rank_consensus` | 0.413 [0.221, 0.634] | 0.335 [0.163, 0.571] | 100.0% |
| 10 | `model_centrality` | 0.413 [0.216, 0.632] | 0.335 [0.162, 0.570] | 100.0% |
| 12 | `hits` | 0.408 [0.220, 0.626] | 0.333 [0.158, 0.571] | 100.0% |
| 13 | `random_candidate` | 0.396 [0.298, 0.491] | 0.348 [0.189, 0.561] | 100.0% |
| 14 | `default_hbos` | 0.390 [0.166, 0.617] | 0.323 [0.145, 0.554] | 100.0% |
| 15 | `default_iforest` | 0.386 [0.191, 0.606] | 0.326 [0.150, 0.565] | 100.0% |
| 16 | `rtm` | 0.330 [0.187, 0.480] | 0.343 [0.159, 0.558] | 100.0% |
| 17 | `ireos` | 0.317 [0.091, 0.570] | 0.351 [0.181, 0.552] | 58.6% |
| 18 | `asi` | 0.316 [0.127, 0.522] | 0.309 [0.126, 0.545] | 100.0% |
| 19 | `default_ocsvm` | 0.307 [0.142, 0.492] | 0.328 [0.160, 0.537] | 100.0% |
| 20 | `laplacian` | 0.275 [0.067, 0.524] | 0.285 [0.103, 0.529] | 100.0% |
| 21 | `asoi` | 0.247 [0.070, 0.467] | 0.296 [0.123, 0.512] | 100.0% |
| 22 | `default_pca` | 0.227 [0.043, 0.460] | 0.287 [0.113, 0.506] | 100.0% |
| 23 | `top_k_stability` | 0.222 [0.046, 0.437] | 0.287 [0.114, 0.512] | 100.0% |
| 24 | `ranking_stability` | 0.218 [0.041, 0.430] | 0.284 [0.114, 0.510] | 100.0% |

## Confirmation

Eight datasets, ten seeds, 4,800 candidates; confirmatory inference for methods
fixed before comparison.

| Rank | Metric | Selection quality (95% CI) | Mean selected AP (95% CI) | Success |
|---:|---|---:|---:|---:|
| 1 | `score_silhouette` | 0.615 [0.407, 0.813] | 0.618 [0.395, 0.810] | 100.0% |
| 2 | `default_ocsvm` | 0.610 [0.397, 0.822] | 0.627 [0.407, 0.826] | 100.0% |
| 3 | `default_lof` | 0.605 [0.391, 0.817] | 0.625 [0.411, 0.817] | 100.0% |
| 4 | `npd` | 0.592 [0.366, 0.819] | 0.634 [0.416, 0.827] | 100.0% |
| 5 | `eag` | 0.588 [0.402, 0.776] | 0.599 [0.385, 0.797] | 100.0% |
| 6 | `excess_mass` | 0.584 [0.347, 0.821] | 0.581 [0.380, 0.761] | 100.0% |
| 7 | `score_calinski_harabasz` | 0.577 [0.378, 0.773] | 0.596 [0.374, 0.796] | 100.0% |
| 8 | `sireos` | 0.556 [0.349, 0.766] | 0.572 [0.347, 0.787] | 100.0% |
| 9 | `mass_volume` | 0.554 [0.333, 0.773] | 0.567 [0.369, 0.747] | 100.0% |
| 10 | `rtm` | 0.543 [0.350, 0.750] | 0.600 [0.389, 0.794] | 100.0% |
| 11 | `model_centrality` | 0.535 [0.387, 0.678] | 0.562 [0.373, 0.720] | 100.0% |
| 12 | `average_rank_consensus` | 0.535 [0.388, 0.681] | 0.562 [0.374, 0.722] | 100.0% |
| 13 | `hits` | 0.533 [0.385, 0.673] | 0.561 [0.375, 0.723] | 100.0% |
| 14 | `score_davies_bouldin` | 0.530 [0.311, 0.748] | 0.566 [0.347, 0.773] | 100.0% |
| 15 | `random_candidate` | 0.528 [0.368, 0.684] | 0.579 [0.376, 0.754] | 100.0% |
| 16 | `default_iforest` | 0.490 [0.354, 0.624] | 0.521 [0.333, 0.693] | 100.0% |
| 17 | `ireos` | 0.467 [0.231, 0.702] | 0.528 [0.300, 0.743] | 86.2% |
| 18 | `top_k_stability` | 0.458 [0.202, 0.715] | 0.577 [0.353, 0.781] | 100.0% |
| 19 | `default_pca` | 0.457 [0.199, 0.720] | 0.576 [0.349, 0.780] | 100.0% |
| 20 | `ranking_stability` | 0.446 [0.173, 0.709] | 0.577 [0.361, 0.776] | 100.0% |
| 21 | `asoi` | 0.401 [0.187, 0.631] | 0.538 [0.330, 0.720] | 100.0% |
| 22 | `default_hbos` | 0.387 [0.166, 0.646] | 0.496 [0.286, 0.709] | 100.0% |
| 23 | `asi` | 0.314 [0.150, 0.515] | 0.497 [0.292, 0.695] | 100.0% |
| 24 | `laplacian` | 0.307 [0.096, 0.552] | 0.495 [0.273, 0.716] | 92.5% |
