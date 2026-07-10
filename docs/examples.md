# Examples

## scikit-learn score polarity

```python
from sklearn.ensemble import IsolationForest
from labelfree.metrics import relative_top_median_score

model = IsolationForest(random_state=0).fit(X)
scores = model.decision_function(X)

value = relative_top_median_score(
    scores,
    score_polarity="higher_is_normal",
)
```

## Repeated-run stability

```python
import numpy as np
from labelfree.metrics import top_k_stability_score

repeated_scores = np.vstack([scores_seed_1, scores_seed_2, scores_seed_3])
stability = top_k_stability_score(repeated_scores, top_fraction=0.05)
```

Rows must score the same samples in the same order.

## IREOS candidate comparison

```python
import numpy as np
from labelfree.metrics import ireos_scores

probabilities = np.vstack([lof_probabilities, knn_probabilities])
values = ireos_scores(X_scaled, probabilities)
```

Rows must contain detector-specific outlier probabilities in `[0, 1]`.
`ireos_scores` selects one shared kernel-width range for a fair comparison.

## Notebooks

- [Metric tuning](https://github.com/OliverHennhoefer/label-free/blob/main/examples/metric_tuning.ipynb)
  uses Optuna to swap label-free objectives across a shared detector pool and
  checks the selected configurations on a labeled holdout.
- [Metric signal analysis](https://github.com/OliverHennhoefer/label-free/blob/main/examples/metric_signal_analysis.ipynb)
  compares label-free rankings with held-out detector quality across datasets
  and random splits.

Install notebook dependencies with:

```bash
python -m pip install "labelfree[examples]"
```

The notebooks contain no cached outputs, so displayed results always correspond
to the installed package and dependencies.
