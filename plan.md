# Label-free metric implementation plan

## Inclusion test

Keep a method only if it is a computable unsupervised metric for anomaly or outlier detector evaluation/model selection.

The current implementation is deprecated and should be ignored for planning. This plan targets a clean replacement API, with no compatibility promise to existing module names, function signatures, or tests.

In scope:
- Input is one of: `(X, scores)`, `(X, score_function)`, a candidate score matrix, or simple synthetic/pseudo samples plus their scores.
- No ground-truth labels are needed.
- The output is a scalar, curve/AUC, or small dict that can rank detectors or hyperparameters.
- Synthetic anomalies are allowed only when generation is a small sampler/corruption step, not a full benchmark, neural generator, or workflow.

Out of scope:
- Complete AutoML/model-selection frameworks, detector libraries, benchmark suites, or training pipelines.
- Metrics that require `y_true`, labeled validation anomalies, event windows, or supervised classifiers trained on real labels.
- Domain-specific image/time-series protocols unless the reusable core metric is separable from the protocol.

## Core metric inventory

| Metric | Paper/access | Public implementation | Method | Implementation plan |
| --- | --- | --- | --- | --- |
| Excess-Mass and Mass-Volume curves | Goix, "How to Evaluate the Quality of Unsupervised Anomaly Detection Algorithms?", arXiv: https://arxiv.org/abs/1607.01152. Earlier EM curve paper: https://proceedings.mlr.press/v38/goix15.html | Author code: https://github.com/ngoix/EMMV_benchmarks. Python package: https://github.com/christian-oleary/emmv | Estimate how compact score level sets are without labels. EM rewards high mass in small volume; MV penalizes volume needed to capture a target mass. Uses Monte Carlo volume scores/samples, so scaling and dimensionality matter. | Reimplement cleanly from paper/reference. Expose explicit orientation: higher EM is better, lower MV is better. |
| IREOS | Marques et al., "Internal Evaluation of Unsupervised Outlier Detection", DOI: https://doi.org/10.1145/3394053. Earlier SSDBM version: https://doi.org/10.1145/2791347.2791352 | Author repos: https://github.com/homarques/ireos-extension and https://github.com/homarques/IREOS-java | Select top outliers and measure how separable they are from the rest of the data over kernel bandwidths, with chance adjustment/p-value. Strong conceptually, but expensive because it estimates separability per point. | Reimplement the smallest faithful variant first. Add classifier variants only if a source requires them. |
| SIREOS | Marques et al., "Similarity-based Unsupervised Evaluation of Outlier Detection", DOI: https://doi.org/10.1007/978-3-031-17849-8_19 | Author repo: https://github.com/homarques/SIREOS | Replaces IREOS classifier separability with a similarity/neighborhood score, making the internal evaluation cheaper and easier to adapt to non-vector similarities. | Reimplement from author repo/paper and document expected orientation. |
| Ranking/top-k stability | Perini et al., "A Ranking Stability Measure for Quantifying the Robustness of Anomaly Detection Methods", PDF: https://imada.sdu.dk/Research/EDML/2020/EDML20_paper_2.pdf | No clear author package found. The metric is simple enough to implement directly. | Refit/rescore on subsamples and compare rankings or top-k sets on overlapping points. Good for hyperparameter tuning when stable rankings are desired, but it measures robustness, not anomaly correctness. | Include as a robustness metric, not as a primary accuracy surrogate. |

## Add next

| Priority | Metric family | Paper/access | Public implementation | Computable input | Short method | Implementation note |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Score-cluster internal indices | Nguyen et al., "An Evaluation Method for Unsupervised Anomaly Detection Algorithms", PDF: https://vjs.ac.vn/jcc/article/download/8455/8709/38290 | UOMS comparison repo: https://github.com/yzhao062/UOMS. Standard indices available in scikit-learn where applicable. | `scores`, plus `n_outliers` or `contamination` | Split scores into predicted outliers and inliers, then compute cluster-validity indices on the two score groups. Include Silhouette, Calinski-Harabasz, Davies-Bouldin, Xie-Beni, and Dunn/Ray-Turi only if formulas stay small. | Add one module, e.g. `score_cluster.py`. Do not add a clustering pipeline. Default split should be explicit, not guessed silently. |
| 2 | ASI / ASOI | Mahmud et al., "ASOI: anomaly separation and overlap index, an internal evaluation metric for unsupervised anomaly detection", open access: https://link.springer.com/article/10.1007/s40747-025-02204-0 | No credible author repo found. GitHub topic mentions ASOI implementations, but no source was clear enough to depend on. | `scores` or predicted normal/anomaly split | ASI measures separation between predicted normal and anomaly groups; ASOI adds distributional overlap, making it closer to the paper's proposed model-selection metric. | Port directly from paper formulas only after formula extraction. Add tests from hand-computed tiny score distributions. |
| 3 | AutoUAD tail metrics: RTM, EAG, NPD | "AutoUAD: Hyper-parameter Optimization for Unsupervised Anomaly Detection", ICLR 2025: https://proceedings.iclr.cc/paper_files/paper/2025/file/bf375a8dabbae40def018957ea68842a-Paper-Conference.pdf | No official implementation found. | `scores` for RTM/EAG; `X` plus `score_function` for NPD | Relative-top-median and expected-anomaly-gap score tail separation in the unlabeled score distribution. Normalized pseudo discrepancy compares scores on validation data against scores on simple pseudo samples such as isotropic Gaussian noise. | Add RTM/EAG first after verifying formulas. Add NPD only as a metric function with a pluggable sampler; no Bayesian optimizer. |
| 4 | Score smoothness / Laplacian Score baseline | He et al., "Laplacian Score for Feature Selection", NIPS 2005: https://proceedings.neurips.cc/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html | Listed by SIREOS repo as a third-party index source: https://github.com/homarques/SIREOS | `(X, scores)` | Treat the anomaly-score vector as a one-dimensional feature and score how well it respects a neighborhood graph on `X`. This is not anomaly-specific, but it is a cheap label-free baseline used around SIREOS. | Low priority. Implement only if the formula is fewer than ~40 lines using scipy/sklearn neighbors. |
| 5 | Consensus score-matrix metrics | Ma et al., "A Large-scale Study on Unsupervised Outlier Model Selection", arXiv: https://arxiv.org/pdf/2104.01422. Related UOMS repo: https://github.com/yzhao062/UOMS | UOMS repo: https://github.com/yzhao062/UOMS | `score_matrix` with shape `(n_models, n_samples)` | ModelCentrality ranks a detector by average rank-agreement with other detectors. HITS hubness and simple consensus-correlation variants estimate model trust from agreement with consensus outlier rankings. | Add only pure score-matrix functions. Do not train candidate models or own a search framework. |

## Edge-case synthetic anomaly metrics

Keep only the small metric core:
- NPD from AutoUAD is acceptable because pseudo samples are just a sampler plus score comparison.
- A simple pseudo-AUC helper is acceptable if it takes `normal_scores` and `synthetic_anomaly_scores` directly.

Do not implement full synthetic anomaly selection/generation frameworks:
- SWSA/AUCp image model-selection workflows are out of scope even if their internal validation scores are useful: https://arxiv.org/abs/2310.10461.
- EAP evaluates auxiliary anomaly quality, not detector quality/hyperparameter quality directly. Keep it out unless this package later adds an auxiliary-anomaly API. Paper/code: https://openreview.net/forum?id=Qq4ge9Qe31 and https://github.com/Lorenzo-Perini/ExpectedAnomalyPosterior.

## Explicit skips

| Method/framework | Why skipped |
| --- | --- |
| MetaOD, ELECT, AutoOD, PyODDS, AutoTSAD, TSB-AutoAD | Full model-selection or AutoML systems, not standalone metrics. |
| PyOD, DeepOD, Anomalib, TimeEval, TSB-AD | Detector/benchmark libraries. Useful for examples, not metrics to implement here. |
| VUS, range-F1, point-adjusted F1, NAB, ROC-AUC, PR-AUC | Need labels or event annotations. |
| Deep early-stopping objectives such as EntropyStop | Training-procedure metric for deep UOD, not a detector-output metric for this package's current API. |
| Outlier exposure / latent outlier exposure / generated-image validation protocols | Too much generation/training workflow. Only accept their precomputed score comparisons if exposed as a simple metric. |

## API shape

Primary API: sklearn-style metric functions over arrays, not estimator scorers.

Examples:
- `relative_top_median_score(scores, *, score_polarity="higher_is_anomalous")`
- `score_cluster_metrics(scores, *, n_outliers=None, contamination=None, score_polarity="higher_is_anomalous")`
- `mass_volume_auc(X, scores, *, score_polarity="higher_is_anomalous", random_state=None)`
- `ranking_stability_score(score_matrix, *, top_k=None, score_polarity="higher_is_anomalous")`

Rules:
- Normalize internally to "higher score means more anomalous".
- Accept only explicit score polarity for raw arrays: `"higher_is_anomalous"` or `"higher_is_normal"`.
- Do not use `"auto"` for raw scores; there is no reliable signal in a naked array.
- Document metric orientation separately from score polarity, because some metrics are naturally lower-is-better, e.g. Mass-Volume AUC or Davies-Bouldin.
- Defer `make_label_free_scorer` or estimator adapters. They are useful only for `GridSearchCV`-style calls that require `scorer(estimator, X, y=None)` and higher-is-better output.

## Implementation status

- 2026-07-07: Reset the package to a fresh `labelfree` skeleton with a minimal `pyproject.toml`, empty examples folder, and a new focused test suite.
- Implemented `score_cluster_metrics(scores, *, n_outliers=None, contamination=None, score_polarity="higher_is_anomalous")` in `labelfree/metrics/score_cluster.py`.
- Source basis: Nguyen et al. propose using internal clustering validation metrics for unsupervised anomaly detector evaluation; UOMS lists this family as an internal evaluation strategy; scikit-learn supplies the standard Silhouette, Calinski-Harabasz, and Davies-Bouldin implementations.
- Metric orientation: Silhouette and Calinski-Harabasz are higher-is-better; Davies-Bouldin and Xie-Beni are lower-is-better.
- Implemented `asi_score(X, scores, ...)` and `asoi_score(X, scores, ...)` in `labelfree/metrics/asoi.py`.
- Source basis: Mahmud, Farou, and Lendak define ASI and ASOI in the open Springer paper. No credible public author implementation was found. ASOI uses the paper's default global weights, `alpha=0.5314` and `beta=0.4686`.
- ASI note: the paper states the ASI value is normalized to `[0, 1]`, but gives Eq. 3 as an unbounded standardized mean difference and does not specify the normalization transform. The implementation returns Eq. 3 exactly instead of inventing an undocumented transform.
- Implemented AutoUAD score-distribution metrics in `labelfree/metrics/autouad.py`: `relative_top_median_score`, `expected_anomaly_gap_score`, and `normalized_pseudo_discrepancy_score`.
- Source basis: Dai and Fan's ICLR 2025 AutoUAD paper gives RTM, EAG, and NPD formulas. No official implementation was found. NPD is exposed over precomputed validation/generated score vectors only; this keeps the metric independent from detector training, data splitting, and pseudo-sample generation.
- Implemented `laplacian_score(X, scores, *, n_neighbors=5, score_polarity="higher_is_anomalous")` in `labelfree/metrics/laplacian.py`.
- Source basis: He, Cai, and Niyogi's Laplacian Score paper defines the graph Rayleigh quotient. SIREOS lists Laplacian Score as a third-party internal index source. The implementation uses a binary k-nearest-neighbor graph to avoid adding a heat-kernel bandwidth knob.
- Implemented consensus score-matrix metrics in `labelfree/metrics/consensus.py`: `model_centrality_scores`, `average_rank_consensus_scores`, and `hits_model_scores`.
- Source basis: Ma et al.'s UOMS paper/repository uses consensus-based internal model-selection measures such as ModelCentrality and HITS. The implementation accepts only an existing `(n_models, n_samples)` score matrix and does not construct detector pools, generate variations, or run model selection workflows.
- Implemented `ranking_stability_score` and `top_k_stability_score` in `labelfree/metrics/stability.py`.
- Source basis: Perini, Galvin, and Vercruyssen define ranking stability over repeated anomaly-score rankings, with author code at https://github.com/Lorenzo-Perini/StabilityRankings_AD. The implementation accepts precomputed repeated score rows and does not own detector refitting or subsampling.
- Implemented `excess_mass_curve`, `excess_mass_auc`, `mass_volume_curve`, and `mass_volume_auc` in `labelfree/metrics/mass_volume.py`.
- Source basis: Goix's arXiv paper and author repository https://github.com/ngoix/EMMV_benchmarks, with cross-check against https://github.com/christian-oleary/emmv. The implementation exposes the computable core over data scores, reference/uniform scores, and support volume; it does not own model scoring or reference-sample generation.
- Implemented `sireos_score(X, scores, ...)` in `labelfree/metrics/sireos.py`.
- Source basis: author repository https://github.com/homarques/SIREOS. The implementation follows the repository's score-weighted average heat-kernel similarity to other points. Lower is better.
- Implemented `ireos_score(X, scores, ...)` in `labelfree/metrics/ireos.py` as a compact IREOS-style RBF separability core over selected candidate outliers.
- IREOS note: the full Java implementation includes max-margin classifier integration, adaptive quadrature, clump handling, and chance adjustment. The current Python function intentionally keeps only the computable metric core suitable for this package API; it is not a full clone of the Java framework.

## Minimal implementation order

1. Done: Define the clean metric API: inputs, orientation, returned scalar/curve fields, and deterministic random-state handling.
2. Done: Add `score_cluster_metrics(scores, *, n_outliers=None, contamination=None)` with Silhouette, Calinski-Harabasz, Davies-Bouldin, Xie-Beni, and a tiny test fixture.
3. Done: Add ASI/ASOI only after paper-formula extraction; include tiny deterministic tests.
4. Done: Add RTM/EAG; add NPD as a pure score-vector metric over validation and generated scores.
5. Done: Add score-matrix consensus metrics only as functions over existing candidate scores.
6. Add one Optuna example that swaps metric objectives without adding any tuning framework.

Skipped: preserving deprecated implementation details. Each replacement metric should be one small module plus focused tests.
