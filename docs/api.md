# API reference

Everything below can be imported from `labelfree`. Metric functions are also
available from `labelfree.metrics`. Every function has a full docstring with
the formula it implements.

## Conventions

- **Score polarity.** Functions that take raw scores accept
  `score_polarity="higher_is_anomalous"` (default) or `"higher_is_normal"`
  and orient scores so that larger values mean more anomalous.
- **Metric direction.** Each function documents whether higher or lower
  values are better. Score polarity never changes that direction.
- **Split size.** Partition metrics need exactly one of `n_outliers` or
  `contamination`; contamination uses `ceil(n_samples * contamination)`.
- **Matrices.** Consensus inputs have shape `(n_models, n_samples)`, stability
  inputs `(n_runs, n_samples)`, and aggregation inputs `(n_metrics,
  n_models)`.
- **Validation.** Inputs are converted to finite float arrays. Invalid shapes,
  non-finite values, mismatched sample counts, invalid fractions, and unknown
  options raise `ValueError`.

## Score helpers

```python
orient_scores(scores, *, score_polarity) -> np.ndarray

unify_scores(
    scores, *, method="gaussian",          # "gaussian", "gamma", "linear"
    score_polarity="higher_is_anomalous",
    inversion="linear",                    # or "log", for higher_is_normal
    baseline=None,
) -> np.ndarray                            # outlier probabilities in [0, 1]
```

## Partition metrics

```python
score_cluster_metrics(
    scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous", indices=None,
) -> dict[str, float]

SCORE_CLUSTER_INDICES  # {"silhouette": "higher", ..., "s_dbw": "lower"}

asi_score(
    X, scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous",
) -> float                                 # higher is better

asoi_score(
    X, scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous", alpha=0.5314, beta=0.4686,
) -> float                                 # higher is better
```

## Score-distribution metrics

```python
relative_top_median_score(
    scores, *, top_fraction=0.05,
    score_polarity="higher_is_anomalous", eps=1e-6,
) -> float                                 # higher is better

expected_anomaly_gap_score(
    scores, *, top_fraction=0.2,
    score_polarity="higher_is_anomalous", eps=1e-6,
) -> float                                 # higher is better

normalized_pseudo_discrepancy_score(
    validation_scores, generated_scores, *,
    score_polarity="higher_is_anomalous", eps=1e-6,
) -> float                                 # higher is better
```

## Feature-space metrics

```python
laplacian_score(
    X, scores, *, n_neighbors=5, kernel_width=None,
    score_polarity="higher_is_anomalous",
) -> float                                 # lower is better

sireos_score(
    X, scores, *, kernel_width=None, kernel_quantile=0.01,
    score_polarity="higher_is_anomalous",
) -> float                                 # lower is better

ireos_score(
    X, outlier_probabilities, *, gamma_max=None, max_clump_size=1,
    penalty_cost=100.0, integration_tol=0.005,
) -> float                                 # higher is better

ireos_scores(
    X, outlier_probability_matrix, *, gamma_max=None, max_clump_size=1,
    penalty_cost=100.0, integration_tol=0.005,
) -> np.ndarray                            # one shared gamma_max

ireos_significance(
    X, outlier_probabilities, *, gamma_max=None, max_clump_size=1,
    penalty_cost=100.0, integration_tol=0.005,
    n_permutations=100, random_state=None,
) -> IREOSSignificance
```

`IREOSSignificance` is a frozen dataclass with `index`, `expected`, `std`,
`adjusted`, `z_score`, `p_value`, and `method` (`"exact"` or
`"monte_carlo"`).

SIREOS needs non-negative oriented scores. IREOS needs probabilities in
`[0, 1]`; automatic `gamma_max` selection needs at least one probability of at
least 0.5.

## Excess-Mass and Mass-Volume

```python
excess_mass_curve(
    scores, reference_scores, *, support_volume, levels=None,
    level_count=1000, score_polarity="higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]

excess_mass_auc(..., em_min=0.9) -> float  # higher is better

mass_volume_curve(
    scores, reference_scores, *, support_volume, alpha_min=0.9,
    alpha_max=0.999, alpha_count=1000, score_polarity="higher_is_anomalous",
) -> tuple[np.ndarray, np.ndarray]

mass_volume_auc(...) -> float              # lower is better

bounding_box_volume(X, *, offset=1e-12) -> float
```

`reference_scores` are detector scores on uniform samples from the support
whose Lebesgue volume is `support_volume`.

## Candidate consensus

All return one higher-is-better value per model.

```python
model_centrality_scores(
    score_matrix, *, similarity="kendall", n_neighbors=None, n_peers=None,
    random_state=None, score_polarity="higher_is_anomalous",
)

udr_scores(
    score_matrix, groups, *, similarity="spearman", n_peers=None,
    random_state=None, score_polarity="higher_is_anomalous",
)

hits_scores(
    score_matrix, *, max_iter=500, tol=1e-10,
    score_polarity="higher_is_anomalous",
)

ens_scores(
    score_matrix, *, similarity="kendall",
    score_polarity="higher_is_anomalous",
)
```

`similarity` is one of `"kendall"`, `"spearman"`, or `"ndcg"`.

## Ranking stability

```python
ranking_stability_score(
    score_matrix, *, contamination, psi=0.8,
    score_polarity="higher_is_anomalous",
) -> float                                 # higher is better, in [0, 1]
```

## Metric aggregation

```python
borda_scores(metric_matrix, *, higher_is_better=True, top_k=None)
robust_borda_scores(metric_matrix, *, higher_is_better=True, top_k=None)
```

`higher_is_better` is one boolean or one per metric row. Both return Borda
points per model; higher is better.
