# labelfree

[![Tests](https://github.com/OliverHennhoefer/label-free/actions/workflows/tests.yml/badge.svg)](https://github.com/OliverHennhoefer/label-free/actions/workflows/tests.yml)
[![Documentation](https://github.com/OliverHennhoefer/label-free/actions/workflows/docs.yml/badge.svg)](https://oliverhennhoefer.github.io/label-free/)
[![PyPI](https://img.shields.io/pypi/v/labelfree.svg)](https://pypi.org/project/labelfree/)

Label-free metrics for comparing unsupervised anomaly detectors and
hyperparameters when labeled anomalies are unavailable.

## Install

```bash
pip install labelfree
```

## Quick start

```python
import numpy as np
from labelfree import model_centrality_scores, score_cluster_metrics

scores = [0.1, 0.2, 0.3, 3.8, 4.1]
metrics = score_cluster_metrics(scores, n_outliers=2)
print(metrics["silhouette"])  # higher is better

score_matrix = np.vstack([scores, [0.2, 0.1, 0.4, 3.0, 5.0], [4.0, 0.1, 0.2, 0.3, 0.1]])
print(model_centrality_scores(score_matrix))  # one value per candidate
```

## Metrics

| Family | Functions |
| --- | --- |
| Score clusters (Nguyen et al.) | `score_cluster_metrics`: silhouette, Calinski-Harabasz, Davies-Bouldin, Xie-Beni, R², RMSSTD, Hubert Γ, I-index, Dunn, S_Dbw |
| ASI / ASOI (Mahmud et al.) | `asi_score`, `asoi_score` |
| AutoUAD (Dai and Fan) | `relative_top_median_score`, `expected_anomaly_gap_score`, `normalized_pseudo_discrepancy_score` |
| Laplacian Score (He et al.) | `laplacian_score` |
| IREOS (Marques et al.) | `ireos_score`, `ireos_scores`, `ireos_significance` |
| SIREOS (Marques et al.) | `sireos_score` |
| Excess-Mass / Mass-Volume (Goix) | `excess_mass_curve`, `excess_mass_auc`, `mass_volume_curve`, `mass_volume_auc` |
| Consensus (Ma et al.) | `model_centrality_scores` (MC, MCS, KNN), `udr_scores`, `hits_scores`, `ens_scores` |
| Ranking stability (Perini et al.) | `ranking_stability_score` |
| Helpers | `unify_scores` (Kriegel et al.), `borda_scores` and `robust_borda_scores` (Goswami et al.) |

Every implementation is tested against its paper and, where available, a
transcription of the authors' code. Functions that take raw scores accept
`score_polarity="higher_is_normal"` for APIs such as scikit-learn's
`decision_function`.

Label-free metrics are model-selection signals, not substitutes for labeled
evaluation. Check each metric's assumptions and direction before comparing
results.

See the [documentation](https://oliverhennhoefer.github.io/label-free/) for
metric selection, API details, and examples.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
```

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Released
under the [MIT License](LICENSE).
