# Choose a metric

| Family | Functions | Input | Better | Main assumption |
| --- | --- | --- | --- | --- |
| Score clusters | `score_cluster_metrics` | scores + split size | mixed | good scores form two separated groups |
| ASI / ASOI | `asi_score`, `asoi_score` | features + scores + split size | higher | predicted anomalies separate in feature space |
| AutoUAD | `relative_top_median_score`, `expected_anomaly_gap_score`, `normalized_pseudo_discrepancy_score` | one or two score vectors | higher | useful detectors create a distinct score tail |
| Graph smoothness | `laplacian_score` | features + scores | lower | nearby samples should have similar scores |
| Consensus | `model_centrality_scores`, `average_rank_consensus_scores`, `hits_model_scores` | model-by-sample score matrix | higher | reliable candidates agree with other reliable candidates |
| Stability | `ranking_stability_score`, `top_k_stability_score` | run-by-sample score matrix | higher | robust detectors preserve rankings after refits |
| IREOS | `ireos_score`, `ireos_scores` | features + outlier probabilities | higher | likely outliers should be easy to separate from other samples |
| SIREOS | `sireos_score` | features + scores | lower | high-scored points should be dissimilar to other samples |
| EM / MV | `excess_mass_*`, `mass_volume_*` | data and uniform-reference scores | EM higher; MV lower | good score level sets have small volume |

## Practical boundaries

**Score clusters** are cheap and useful for screening, but can reward an
arbitrary bimodal score distribution. The returned dictionary mixes
higher-is-better and lower-is-better values.

**ASI and ASOI** use the score-induced normal/anomaly partition. Scale features
before comparing models. ASI implements Equation 3 of the paper; the paper says
the value is normalized to `[0, 1]` but does not define an additional
transform, so this implementation does not invent one. ASOI's separation term
is also scale-dependent.

**RTM and EAG** are most comparable within one score convention because RTM is
not invariant to score offsets. **NPD** compares validation scores with scores
on generated reference samples; this package accepts those precomputed vectors
and does not generate or train anything.

**Consensus metrics** need at least two candidates. If most candidates share
the same failure mode, agreement is not evidence of correctness.

**Ranking stability** measures robustness, not accuracy. Rows must be repeated
scores for the same samples.

**IREOS**, **SIREOS**, and **Laplacian Score** depend on feature scaling. IREOS
trains a kernel logistic classifier repeatedly and is intentionally expensive.
Pass probabilities normalized for the detector that produced them; a universal
raw-score transformation would not reproduce the paper. Use `ireos_scores` to
compare candidates under one shared kernel-width range. The optional clump-size
heuristic from the paper is `sqrt(0.05 * len(X))`; choose its integer value
explicitly. IREOS and SIREOS both use quadratic memory for pairwise distances.

**Excess-Mass and Mass-Volume** require scores on a uniform reference sample
over a known support. They become unreliable as Monte Carlo volume estimation
deteriorates in high dimensions.

## Source basis

- Score-cluster indices: [Nguyen et al.](https://vjs.ac.vn/jcc/article/download/8455/8709/38290)
  and the standard scikit-learn metric definitions.
- ASI/ASOI: [Mahmud, Farou, and Lendak](https://link.springer.com/article/10.1007/s40747-025-02204-0).
- AutoUAD: [Dai and Fan, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/bf375a8dabbae40def018957ea68842a-Paper-Conference.pdf).
- Consensus: [Ma et al.](https://arxiv.org/abs/2104.01422) and the
  [UOMS repository](https://github.com/yzhao062/UOMS).
- Stability: [Perini et al.](https://lorenzo-perini.github.io/files/RankingStabilityPaper.pdf)
  and the [author implementation](https://github.com/Lorenzo-Perini/StabilityRankings_AD).
- IREOS: [Marques et al.](https://doi.org/10.1145/3394053) and the
  [author implementation](https://github.com/homarques/ireos-extension).
- SIREOS: [Marques et al.](https://doi.org/10.1007/978-3-031-17849-8_19)
  and the [author implementation](https://github.com/homarques/SIREOS).
- EM/MV: [Goix](https://arxiv.org/abs/1607.01152) and the
  [author implementation](https://github.com/ngoix/EMMV_benchmarks).
- Laplacian Score: [He, Cai, and Niyogi](https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html).
