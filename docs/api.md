# API reference

Import public functions from `labelfree.metrics`.

## Partition metrics

```python
score_cluster_metrics(
    scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous"
) -> dict[str, float]

asi_score(
    X, scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous"
) -> float

asoi_score(
    X, scores, *, n_outliers=None, contamination=None,
    score_polarity="higher_is_anomalous",
    alpha=0.5314, beta=0.4686
) -> float
```

Pass exactly one of `n_outliers` and `contamination`. Contamination uses
`ceil(n_samples * contamination)`.

## Score-distribution metrics

```python
relative_top_median_score(
    scores, *, top_fraction=0.05,
    score_polarity="higher_is_anomalous", eps=1e-6
) -> float

expected_anomaly_gap_score(
    scores, *, top_fraction=0.2,
    score_polarity="higher_is_anomalous", eps=1e-6
) -> float

normalized_pseudo_discrepancy_score(
    validation_scores, generated_scores, *,
    score_polarity="higher_is_anomalous", eps=1e-6
) -> float
```

All three return higher-is-better scalars.

## Feature-space metrics

```python
laplacian_score(
    X, scores, *, n_neighbors=5,
    score_polarity="higher_is_anomalous"
) -> float

sireos_score(
    X, scores, *, score_polarity="higher_is_anomalous",
    kernel_width=None, kernel_quantile=0.01
) -> float

ireos_score(
    X, outlier_probabilities, *, gamma_max=None,
    max_clump_size=1, penalty_cost=100.0,
    integration_tol=0.005
) -> float

ireos_scores(
    X, outlier_probability_matrix, *, gamma_max=None,
    max_clump_size=1, penalty_cost=100.0,
    integration_tol=0.005
) -> np.ndarray
```

IREOS is higher-is-better; SIREOS and Laplacian Score are lower-is-better.
IREOS requires detector-specific outlier probabilities in `[0, 1]`, not raw
scores. The plural function uses one shared automatically selected `gamma_max`
for all rows. Automatic selection requires at least one probability above
`0.5`; otherwise, pass `gamma_max` explicitly. If `kernel_width` is omitted,
SIREOS uses the requested quantile
of nonzero pairwise distances.

## Candidate consensus

```python
model_centrality_scores(score_matrix, *, score_polarity="higher_is_anomalous")
average_rank_consensus_scores(score_matrix, *, score_polarity="higher_is_anomalous")
hits_model_scores(
    score_matrix, *, score_polarity="higher_is_anomalous",
    max_iter=100, tol=1e-10
)
```

`score_matrix` has shape `(n_models, n_samples)`. Each function returns one
higher-is-better value per model.

## Ranking stability

```python
ranking_stability_score(
    score_matrix, *, contamination, psi=0.8,
    score_polarity="higher_is_anomalous"
) -> float

top_k_stability_score(
    score_matrix, *, top_k=None, top_fraction=None,
    score_polarity="higher_is_anomalous"
) -> float
```

`score_matrix` has shape `(n_runs, n_samples)`. Pass exactly one of
`top_k` and `top_fraction` to top-k stability.

## Excess-Mass and Mass-Volume

```python
excess_mass_curve(scores, reference_scores, *, support_volume, ...)
excess_mass_auc(scores, reference_scores, *, support_volume, ...)
mass_volume_curve(scores, reference_scores, *, support_volume, ...)
mass_volume_auc(scores, reference_scores, *, support_volume, ...)
bounding_box_volume(X, *, offset=1e-12) -> float
```

Curve functions return `(axis, values)`. `reference_scores` are detector
scores on uniform samples from the support whose Lebesgue volume is
`support_volume`. `bounding_box_volume` is a convenience for an
axis-aligned support; it does not generate the reference samples.

## Validation

Inputs are converted to finite floating-point arrays. Invalid dimensions,
non-finite values, inconsistent sample counts, invalid fractions, and unknown
score polarities raise `ValueError`.
