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
from labelfree.metrics import score_cluster_metrics

scores = [0.1, 0.2, 0.3, 3.8, 4.1]
metrics = score_cluster_metrics(scores, n_outliers=2)
print(metrics["silhouette"])  # higher is better
```

The package includes score-cluster, ASI/ASOI, AutoUAD, Laplacian,
consensus, ranking-stability, IREOS, SIREOS, and Excess-Mass/Mass-Volume metrics.
Every raw-score API requires an explicit score polarity when larger values
mean more normal.

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
