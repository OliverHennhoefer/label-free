# Examples

## scikit-learn score polarity

```python
from sklearn.ensemble import IsolationForest
from labelfree import relative_top_median_score

model = IsolationForest(random_state=0).fit(X)
scores = model.decision_function(X)

value = relative_top_median_score(scores, score_polarity="higher_is_normal")
```

## Outlier probabilities for IREOS and SIREOS

```python
from labelfree import ireos_significance, sireos_score, unify_scores

probabilities = unify_scores(scores, score_polarity="higher_is_normal")

similarity = sireos_score(X_scaled, probabilities)       # lower is better
result = ireos_significance(X_scaled, probabilities)
print(result.index, result.adjusted, result.p_value)
```

`unify_scores` uses Gaussian scaling by default, which maps every score at or
below the mean to 0. Use `method="linear"` to keep weight on every sample.

## IREOS candidate comparison

```python
import numpy as np
from labelfree import ireos_scores, unify_scores

probabilities = np.vstack(
    [unify_scores(lof_scores, baseline=1.0), unify_scores(knn_scores)]
)
values = ireos_scores(X_scaled, probabilities)
```

`ireos_scores` selects one shared kernel-width range for a fair comparison.

## Consensus over a model pool

```python
import numpy as np
from labelfree import ens_scores, hits_scores, model_centrality_scores, udr_scores

score_matrix = np.vstack([model.decision_function(X) for model in models])
families = [type(model).__name__ for model in models]

mc = model_centrality_scores(score_matrix, score_polarity="higher_is_normal")
knn_mc = model_centrality_scores(
    score_matrix, n_neighbors=3, score_polarity="higher_is_normal"
)
udr = udr_scores(score_matrix, families, score_polarity="higher_is_normal")
hits = hits_scores(score_matrix, score_polarity="higher_is_normal")
ens = ens_scores(score_matrix, score_polarity="higher_is_normal")
```

## Repeated-run stability

```python
import numpy as np
from labelfree import ranking_stability_score

rng = np.random.default_rng(0)
runs = []
for _ in range(50):
    subsample = rng.choice(len(X_train), size=len(X_train) // 2, replace=False)
    runs.append(make_detector().fit(X_train[subsample]).decision_function(X_eval))

stability = ranking_stability_score(
    np.vstack(runs), contamination=0.05, score_polarity="higher_is_normal"
)
```

## Combining metrics

```python
import numpy as np
from labelfree import robust_borda_scores

metric_matrix = np.vstack([mc, hits, ens, mass_volume_values])
points = robust_borda_scores(
    metric_matrix, higher_is_better=[True, True, True, False], top_k=5
)
best_model = models[int(np.argmax(points))]
```

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
