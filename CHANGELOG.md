# Changelog

## 1.1.0 - 2026-11-22

This release fixes metric implementation errors found in an audit for
degenerate inputs (ties, constant scores, offsets, duplicate rows) and metric
direction.

### Behavior changes

Some inputs that used to return a misleading value now raise `ValueError`:

- constant scores, or scores tied across the whole `n_outliers` cutoff, in
  `asi_score`, `asoi_score`, and `score_cluster_metrics`;
- `relative_top_median_score` with a negative median oriented score;
- non-boolean `higher_is_better` values in `borda_scores` and
  `robust_borda_scores`;
- `bounding_box_volume` with a constant feature;
- `excess_mass_auc` with custom levels whose curve starts at or below `em_min`.

Results also change for ASI, ASOI, and the score-cluster indices whenever
scores tie at the cutoff, because every tied score now joins the anomaly
group.

### Metric fixes

- ASI, ASOI, and the score-cluster indices put every score tied with the
  `n_outliers` cutoff in the anomaly group, so results no longer depend on row
  order.
- Constant scores no longer receive the best possible Davies-Bouldin and
  RMSSTD values; they raise because they do not split the data.
- Calinski-Harabasz is `inf` for a split with no within-cluster spread;
  scikit-learn's fallback of 1.0 ranked a perfect split below noisy ones.
- EAG centers the scores before its running sums, so shifting scores by a
  large constant no longer changes the result.
- Laplacian Score rescales the scores before centering, so constant scores
  are always `inf` rather than sometimes 0, the best value.
- RTM rejects scores with a negative median, where the ratio reverses
  direction.
- The automatic IREOS `gamma_max` search skips candidates that duplicate
  another row of `X`, with a warning, instead of failing.
- `borda_scores` and `robust_borda_scores` require boolean directions, so
  values such as `"lower"` or `-1` are no longer read as higher-is-better.
- `excess_mass_auc` raises when custom levels start with the curve already at
  or below `em_min`, which returned an area of 0.
- `bounding_box_volume` raises for constant features instead of returning
  only its `offset`, which inflated Excess-Mass areas by orders of magnitude.
- The example notebooks score fitted rows with PyOD's leave-one-out
  `decision_scores_` instead of `decision_function`, which counts every row as
  its own neighbor. The metric comparison notebook also shifts scores to be
  non-negative for RTM and draws Excess-Mass feature subsets only from
  non-constant features.

### Dependencies

- Minimum versions are now numpy 1.26, scipy 1.11.2, and scikit-learn 1.3.1,
  the oldest releases that install and pass on Python 3.12. The previous
  bounds (numpy 1.24, scipy 1.10) have no Python 3.12 wheels.

### Documentation and CI

- The README is organized around choosing a metric, with a runnable quick
  start that ranks four scikit-learn detectors.
- CI tests the minimum dependencies, weekly pre-releases of NumPy, SciPy, and
  scikit-learn, and macOS and Windows on Python 3.14, and builds and
  smoke-tests the package on every pull request. Dependabot keeps the
  GitHub Actions current.
- The documentation deploy works again with the current GitHub Pages
  actions, and pull requests now check that the site builds.

## 1.0.0 - 2026-07-10

- First stable functional API for label-free anomaly-detector metrics, with
  every public name importable from `labelfree`.
- Score-cluster indices now cover all ten indices of Nguyen et al.: R²,
  RMSSTD, Hubert Γ, I-index, Dunn, and S_Dbw join silhouette,
  Calinski-Harabasz, Davies-Bouldin, and Xie-Beni. `SCORE_CLUSTER_INDICES`
  lists their directions and `indices=` selects a subset.
- Consensus strategies from Ma et al.: UDR (`udr_scores`), selective-ensemble
  ENS (`ens_scores`), sampled MCS and K-nearest-neighbor centrality
  (Goswami et al.) in `model_centrality_scores`, and Kendall, Spearman, or
  NDCG similarity.
- `hits_scores` uses the published `1 / rank` edge weights with the top
  outlier at rank 1.
- `ireos_significance` adjusts IREOS for chance and reports a z-score and a
  permutation p-value against random solutions.
- IREOS follows the authors' `gamma_max` search exactly and solves kernel
  logistic regression with Newton's method in the kernel's feature space,
  which stays stable for small kernel widths.
- SIREOS weights samples by `s / sum(s)` as in the paper and authors' code
  and rejects negative scores instead of shifting them.
- Laplacian Score supports heat-kernel weights and excludes self-neighbors
  correctly for duplicate rows.
- `unify_scores` implements the score unification of Kriegel et al.
- `borda_scores` and `robust_borda_scores` aggregate several metrics as in
  Goswami et al.
- Removed the unsourced `average_rank_consensus_scores` and
  `top_k_stability_score`; `hits_model_scores` is now `hits_scores`, and
  `labelfree.utils` is private.
- Reference tests against the authors' implementations (UOMS, IREOS,
  SIREOS, EMMV, ranking stability), ELKI formulas, and the `s_dbw` package.
- Zensical documentation, two example notebooks stored with their outputs
  (metric comparison and label-free model selection), CI, and trusted PyPI
  publishing.
