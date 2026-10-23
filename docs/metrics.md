# Choose a metric

Every metric here is an internal criterion: it uses only the data, the
detector scores, or several candidates' scores. None of them needs labels, and
none of them is a substitute for labeled evaluation. Ma et al. (2023) found
that no internal strategy reliably beats a well-configured default, so treat
these values as evidence and check them across splits.

| Family | Functions | Input | Better | Assumption |
| --- | --- | --- | --- | --- |
| Score clusters | `score_cluster_metrics` | scores + split size | per index | good scores split into two separated groups |
| ASI / ASOI | `asi_score`, `asoi_score` | features + scores + split size | higher | predicted anomalies separate in feature space |
| AutoUAD | `relative_top_median_score`, `expected_anomaly_gap_score`, `normalized_pseudo_discrepancy_score` | one or two score vectors | higher | useful detectors create a distinct score tail |
| Laplacian Score | `laplacian_score` | features + scores | lower | nearby samples have similar scores |
| IREOS | `ireos_score`, `ireos_scores`, `ireos_significance` | features + outlier probabilities | higher | outliers are easy to separate from the rest |
| SIREOS | `sireos_score` | features + non-negative scores | lower | outliers are dissimilar to the rest |
| EM / MV | `excess_mass_*`, `mass_volume_*` | data and uniform-reference scores | EM higher, MV lower | good normality level sets have small volume |
| Consensus | `model_centrality_scores`, `udr_scores`, `hits_scores`, `ens_scores` | model-by-sample score matrix | higher | good candidates agree with other good candidates |
| Stability | `ranking_stability_score` | run-by-sample score matrix | higher | robust detectors keep their ranking across refits |

Two helpers support these metrics. `unify_scores` turns raw scores into
outlier probabilities (Kriegel et al., 2011), which IREOS expects and which
make signed scores usable for SIREOS. `borda_scores` and
`robust_borda_scores` combine several metrics into one model ranking
(Goswami et al., 2023).

## Score clusters

`score_cluster_metrics` treats the scores as a one-dimensional dataset, labels
the top `n_outliers` as one cluster, and returns ten internal clustering
indices, as Nguyen et al. proposed and Ma et al. benchmarked:

| Index | Better | Definition on the split |
| --- | --- | --- |
| `silhouette` | higher | scikit-learn silhouette |
| `calinski_harabasz` | higher | scikit-learn Calinski-Harabasz |
| `davies_bouldin` | lower | scikit-learn Davies-Bouldin |
| `xie_beni` | lower | `SSW / (n * d(c_normal, c_anomaly)^2)` |
| `r_squared` | higher | `(SST - SSW) / SST` |
| `rmsstd` | lower | `sqrt(SSW / (n - 2))` |
| `hubert_gamma` | higher | mean of `d(x_i, x_j) * d(c_i, c_j)` over pairs |
| `i_index` | higher | `(0.5 * E_1 / E_2 * d(c_normal, c_anomaly))^2` |
| `dunn` | higher | min inter-cluster distance / max diameter |
| `s_dbw` | lower | Halkidi and Vazirgiannis Scat + Dens_bw |

`SCORE_CLUSTER_INDICES` maps each name to its direction. In one dimension with
a fixed split size, `xie_beni`, `r_squared`, and `calinski_harabasz` rank
candidates identically. All indices can reward an arbitrary bimodal score
distribution.

## ASI and ASOI

Both use the score-induced partition in feature space, so scale features
first. ASI implements Equation 3 of the paper. The paper mentions a
normalization to `[0, 1]` without defining it, so none is applied. The ASOI
separation term is scale-dependent.

## AutoUAD

RTM and EAG are computed on one score vector. RTM is not invariant to score
offsets, so compare it within one score convention. NPD compares scores on
held-out data with scores on generated Gaussian samples. The package accepts
both score vectors and does not generate or train anything.

## Laplacian Score

The default uses the paper's 0/1 k-nearest-neighbor weights. Pass
`kernel_width` for heat-kernel weights. The score is invariant to score scale
and sign.

## IREOS

IREOS trains a kernel logistic regression classifier for every candidate
outlier and kernel width, so it is expensive and uses quadratic memory. Pass
outlier probabilities, for example from `unify_scores`, not raw scores. Use
`ireos_scores` to compare candidates under one shared `gamma_max`. The paper's
clump-size heuristic is `round(sqrt(0.05 * n_samples))`.

`ireos_significance` adjusts the index for chance, `(I - E[I]) / (1 - E[I])`,
and reports a z-score and p-value against solutions that assign the same
probabilities to random samples. For `max_clump_size=1` the null moments are
exact but need one separability curve per sample. For larger clumps they are
estimated by permutation.

## SIREOS

SIREOS weights each sample by `s_j / sum(s)`, so the oriented scores must be
non-negative. Map signed scores, such as scikit-learn `decision_function`
output, with `unify_scores` first. The kernel width defaults to the 1%
quantile of non-zero pairwise distances, as in the authors' code.

## Excess-Mass and Mass-Volume

Both need detector scores on a uniform reference sample over a support of
known volume. They become unreliable as Monte Carlo volume estimates degrade
in high dimensions. Goix suggests averaging over random feature subsets of
size 5 there, which requires refitting the detector and so stays with the
caller.

## Consensus

Consensus strategies need at least two candidates and compare rankings, not
raw scores. If most candidates share a failure mode, agreement is not
evidence of correctness.

- `model_centrality_scores` (MC) is the mean similarity to all other models.
  `n_peers` gives the sampled MCS variant and `n_neighbors` the
  K-nearest-neighbor centrality of Goswami et al.
- `udr_scores` is the median similarity to models of the same group, for
  example other configurations of the same detector.
- `hits_scores` returns hub scores of a model-sample graph with edge weights
  `1 / rank`, so agreement on the top outliers counts most.
- `ens_scores` is the similarity to a pseudo ground truth built by greedy
  selective ensembling.

Similarity can be Kendall's tau, Spearman's rho, or NDCG. The defaults follow
each method's source.

## Ranking stability

Ranking stability measures robustness, not accuracy. Each row must hold
scores for the same samples from one refit, for example on a random
subsample of the training data, as in Perini et al.

## Combining metrics

`borda_scores` sums `N - position` points per metric. `top_k` gives partial
Borda. `robust_borda_scores` first drops the metrics whose empirical influence
on the consensus is largest, following Goswami et al.

## Sources and verification

Each implementation is tested against the paper's formula and, where the
authors published code, against a transcription of that code.

| Metric | Source | Verified against |
| --- | --- | --- |
| Score clusters | [Nguyen et al.](https://vjs.ac.vn/jcc/article/download/8455/8709/38290); [Ma et al.](https://kdd.org/exploration_files/p19-InternalEvaluationOD_Ma2023.pdf) | UOMS `ten_metrics_mat.py`, scikit-learn, `s_dbw` (Halkidi) |
| ASI / ASOI | [Mahmud, Farou, and Lendak](https://link.springer.com/article/10.1007/s40747-025-02204-0) | paper equations |
| AutoUAD | [Dai and Fan, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/bf375a8dabbae40def018957ea68842a-Paper-Conference.pdf) | Algorithm 1 |
| Laplacian Score | [He, Cai, and Niyogi](https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html) | dense matrix definition |
| IREOS | [Marques et al.](https://doi.org/10.1145/3394053) | [author implementation](https://github.com/homarques/ireos-extension) |
| SIREOS | [Marques et al.](https://doi.org/10.1007/978-3-031-17849-8_19) | [author implementation](https://github.com/homarques/SIREOS) |
| EM / MV | [Goix](https://arxiv.org/abs/1607.01152) | [author implementation](https://github.com/ngoix/EMMV_benchmarks) |
| MC, MCS, UDR, HITS, ENS | [Ma et al.](https://arxiv.org/abs/2104.01422); [Lin et al.](https://arxiv.org/abs/1906.06034); [Duan et al.](https://arxiv.org/abs/1905.12614); [Rayana and Akoglu](https://arxiv.org/abs/1501.01924) | [UOMS](https://github.com/yzhao062/UOMS), Algorithm 1 |
| KNN centrality, Borda | [Goswami et al., ICLR 2023](https://arxiv.org/abs/2210.01078) | paper definitions |
| Ranking stability | [Perini et al.](https://lorenzo-perini.github.io/files/RankingStabilityPaper.pdf) | [author implementation](https://github.com/Lorenzo-Perini/StabilityRankings_AD) |
| Score unification | [Kriegel et al.](https://doi.org/10.1137/1.9781611972818.2) | ELKI scaling classes |

### Deliberate deviations from author code

- **EM curve:** the supremum also includes the whole support (`1 - t * V`),
  which the paper's definition admits but Goix's code omits. When the curve
  never drops to `em_min`, the full level range is integrated; Goix drops
  the last level.
- **HITS:** UOMS ranks scores in ascending order before taking `1 / rank`,
  which puts the largest weights on the most normal samples. The paper
  defines `r_ij` as the position in the ranked list of outliers, so the top
  outlier gets weight 1 here.
- **MCS:** the paper averages similarities over the sampled peers; the UOMS
  script reuses UDR's median.
- **S_Dbw:** the original Halkidi and Vazirgiannis definition with mean
  centers, instead of the `s_dbw` package defaults that UOMS used.
- **RMSSTD:** the UOMS script scales by the data dimension; here `P = 1`
  because the index is computed on one-dimensional scores. This does not
  change rankings within a dataset.
- **Ties:** stability and consensus metrics give tied scores their average
  rank; author code breaks ties by sort order.
