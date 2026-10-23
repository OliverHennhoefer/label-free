# Changelog

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
- Zensical documentation, examples, CI, and trusted PyPI publishing.
