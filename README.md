<h1 align="center">labelfree</h1>

<p align="center">
  <strong>Compare and select anomaly detectors when you have no labeled anomalies.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/labelfree/"><img src="https://img.shields.io/pypi/v/labelfree.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/labelfree/"><img src="https://img.shields.io/pypi/pyversions/labelfree.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/OliverHennhoefer/label-free/actions/workflows/tests.yml"><img src="https://github.com/OliverHennhoefer/label-free/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://oliverhennhoefer.github.io/label-free/"><img src="https://github.com/OliverHennhoefer/label-free/actions/workflows/docs.yml/badge.svg" alt="Documentation"></a>
  <a href="https://github.com/OliverHennhoefer/label-free/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/labelfree.svg" alt="License"></a>
</p>

<p align="center">
  <a href="https://oliverhennhoefer.github.io/label-free/">Documentation</a> ·
  <a href="https://oliverhennhoefer.github.io/label-free/metrics/">Metric guide</a> ·
  <a href="https://oliverhennhoefer.github.io/label-free/api/">API reference</a> ·
  <a href="https://github.com/OliverHennhoefer/label-free/tree/main/examples">Examples</a>
</p>

---

**Choosing an anomaly detector usually means scoring it against labeled anomalies, which you rarely have.**
Label-free metrics judge a detector by the structure of its scores, by the data
it was fitted on, or by how well it agrees with other candidates. `labelfree`
collects the published metrics behind one consistent API, each one tested
against its paper.

It works with any detector that produces one score per row, such as
[PyOD](https://pyod.readthedocs.io/) models, scikit-learn estimators or your own.

## Installation

`labelfree` requires Python 3.12 or newer and depends only on NumPy, SciPy and scikit-learn.

```bash
pip install labelfree
```

<details>
<summary><strong>Optional extras</strong></summary>

| Extra | Adds |
| --- | --- |
| `examples` | PyOD, Optuna, pandas, matplotlib and `oddball` datasets for the example notebooks |
| `dev` | pytest, Ruff and pre-commit |
| `docs` | The documentation builder |

</details>

## Quick start

Score the same unlabeled batch with several candidates, then rank them:

```python
import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

import labelfree as lf

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(size=(500, 4)), rng.normal(4, 1, size=(25, 4))])  # unlabeled

candidates = {
    "iforest": IsolationForest(random_state=0),
    "iforest-64": IsolationForest(max_samples=64, random_state=0),
    "ocsvm": OneClassSVM(nu=0.05),
    "elliptic": EllipticEnvelope(random_state=0),
}
# scikit-learn's score_samples is higher for more normal points
scores = np.vstack([m.fit(X).score_samples(X) for m in candidates.values()])

agreement = lf.model_centrality_scores(scores, score_polarity="higher_is_normal")
separation = [
    lf.score_cluster_metrics(s, contamination=0.05, score_polarity="higher_is_normal")[
        "silhouette"
    ]
    for s in scores
]
for name, a, s in zip(candidates, agreement, separation, strict=True):
    print(f"{name:10} centrality={a:.2f}  silhouette={s:.2f}")  # higher is better
# iforest    centrality=0.83  silhouette=0.76
# iforest-64 centrality=0.80  silhouette=0.80
# ocsvm      centrality=0.77  silhouette=0.46
# elliptic   centrality=0.86  silhouette=0.93
```

Every function that takes raw scores assumes that higher means more anomalous
unless you pass `score_polarity="higher_is_normal"`.

## Choose a metric

Pick by the evidence you have. The [metric guide](https://oliverhennhoefer.github.io/label-free/metrics/) covers assumptions and directions.

| You have | Metrics |
| --- | --- |
| One score vector per candidate | `score_cluster_metrics` (silhouette, Calinski-Harabasz, Davies-Bouldin, Xie-Beni, R², RMSSTD, Hubert Γ, I-index, Dunn, S_Dbw) · `relative_top_median_score`, `expected_anomaly_gap_score` (AutoUAD) |
| Scores and the feature matrix | `asi_score`, `asoi_score` · `laplacian_score` · `sireos_score` · `ireos_score`, `ireos_scores`, `ireos_significance` |
| Scores from several candidates on the same rows | `model_centrality_scores` (MC, MCS, KNN) · `udr_scores` · `hits_scores` · `ens_scores` |
| Scores on normal validation data and on generated points | `normalized_pseudo_discrepancy_score` (AutoUAD) |
| Scores from repeated refits | `ranking_stability_score` |
| Scores on data and on uniform reference samples | `excess_mass_curve`, `excess_mass_auc`, `mass_volume_curve`, `mass_volume_auc` |
| Several metric rankings to combine | `borda_scores`, `robust_borda_scores` · `unify_scores` turns raw scores into outlier probabilities |

<details>
<summary><strong>Sources</strong></summary>

| Family | Paper |
| --- | --- |
| Score clusters | Nguyen et al. |
| ASI / ASOI | Mahmud et al. |
| AutoUAD | Dai and Fan |
| Laplacian Score | He et al. |
| IREOS, SIREOS | Marques et al. |
| Excess-Mass / Mass-Volume | Goix |
| Consensus | Ma et al. |
| Ranking stability | Perini et al. |
| Score unification | Kriegel et al. |
| Robust Borda | Goswami et al. |

Each implementation is tested against its paper and, where available, a
transcription of the authors' code.

</details>

## Examples

Two executed notebooks use real benchmark data and PyOD detectors. Install their dependencies with `pip install "labelfree[examples]"`.

- [Metric comparison](https://github.com/OliverHennhoefer/label-free/blob/main/examples/metric_comparison.ipynb) computes every metric family for a pool of detectors, then uses the held-back labels to check how well each metric ranked them.
- [Model selection](https://github.com/OliverHennhoefer/label-free/blob/main/examples/model_selection.ipynb) tunes four detector families with Optuna on a label-free objective, picks one with robust Borda counting, and measures the pick on a labeled test split.

## Scope

> [!IMPORTANT]
> Label-free metrics are model-selection signals, not substitutes for labeled
> evaluation. Large empirical studies have found that no single internal metric
> reliably beats a sensible default detector across datasets. Each metric
> assumes something about what anomalies look like. Check that assumption and
> the metric's direction before you trust its ranking, and confirm an important
> choice on labeled data when you can.

## Citation

If you use `labelfree` in academic work, please cite the software:

```bibtex
@software{Hennhoefer_labelfree,
  author = {Hennhöfer, Oliver},
  title = {{labelfree: Label-free metrics for unsupervised anomaly detector evaluation}},
  url = {https://github.com/OliverHennhoefer/label-free},
  license = {MIT}
}
```

Please also cite the papers behind the metrics you use.

## Project

Read the [documentation](https://oliverhennhoefer.github.io/label-free/), browse
the [changelog](https://github.com/OliverHennhoefer/label-free/blob/main/CHANGELOG.md),
or report a problem in the [issue tracker](https://github.com/OliverHennhoefer/label-free/issues).
Contributions are welcome; start with the [contributing guide](CONTRIBUTING.md).
To develop locally:

```bash
python -m pip install -e ".[dev]"
pytest
```

`labelfree` is distributed under the [MIT License](LICENSE).
