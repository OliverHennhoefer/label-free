# labelfree

`labelfree` provides small, functional metrics for comparing unsupervised
anomaly detectors without ground-truth labels. It evaluates existing score
vectors or candidate score matrices; it does not train detectors or own a
model-selection workflow.

## Install

```bash
pip install labelfree
```

Python 3.12 or newer is required.

## First metric

```python
from labelfree.metrics import score_cluster_metrics

scores = [0.1, 0.2, 0.3, 3.8, 4.1]
result = score_cluster_metrics(scores, n_outliers=2)

print(result["silhouette"])       # higher is better
print(result["davies_bouldin"])   # lower is better
```

The split size is always explicit: pass either `n_outliers` or
`contamination`.

## Score polarity

All raw-score functions normalize scores internally so larger values mean more
anomalous. The default is `score_polarity="higher_is_anomalous"`. Use
`"higher_is_normal"` for APIs such as scikit-learn's
`IsolationForest.decision_function`.

Score polarity is separate from metric direction. For example, high input
scores can mean anomalous while a lower Mass-Volume AUC is still better.

## Use metrics as evidence, not labels

Label-free metrics encode assumptions: separation, smoothness, agreement,
compact level sets, or robustness. A detector can satisfy one assumption and
still be wrong. Compare multiple candidates under the same preprocessing,
inspect sensitivity across splits, and use labeled evaluation when reliable
labels become available.
