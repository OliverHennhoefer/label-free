# Changelog

## Unreleased

- ASI, ASOI, and the score-cluster indices put every score tied with the
  `n_outliers` cutoff in the anomaly group, so results no longer depend on row
  order. Scores tied across the whole cutoff, such as constant scores, raise
  `ValueError` instead of receiving best-possible Davies-Bouldin and RMSSTD
  values.
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
- The example notebooks score fitted rows with PyOD's leave-one-out
  `decision_scores_` instead of `decision_function`, which counts every row as
  its own neighbor.

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
