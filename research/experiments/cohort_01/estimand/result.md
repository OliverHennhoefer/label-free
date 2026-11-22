# Endpoint agreement can reverse under an unknown monotone anomaly risk

**Cohort decision, 2026-11-22.** Withhold the selector and its matched control. The registered prediction required a strict change in the best conservative choice and a reduction of at least 0.01 in worst-case population AP regret. The improvement from a strict switch was zero across all 258,840 pairs in the specified six-point experiment. The two methods also chose identical candidates on both seeds of all four previously open sources. This arm received no fresh development grade. [Original prediction and scope](prediction.md)

## Exhaustive known-truth test

The six equally probable feature values were −3, −2, −1, 1, 2, 4. Radius gave a strict severity ordering. The seven registered dyadic budgets reduced to four distinct nested sets on this support, containing 1, 2, 3 and 6 rows. Each pair of the 720 possible candidate rankings defined one decision problem. We evaluated every pair on 1,771 soft risk profiles on the registered grid and maximized pairwise AP regret over the continuous simplex by the face calculation in the prediction. The bottom three rows had identical risk profiles at every calibration. Accounting for the multiplicity of each ranking reduced the exact work to 7,140 pairs of distinct ranking types, plus pairs equivalent at all calibrations.

| Observation over 258,840 unordered pairs | Count |
| --- | ---: |
| Opposite strict minimax choices from hard and soft calibrations | 0 |
| Ties under the hard-cutoff criterion | 4,212 |
| Ties under the full calibration criterion | 4,140 |
| Hard ties resolved by the full criterion | 72 |
| Hard-endpoint dominance, strict at one or more endpoints, reversed at an admissible mixture | 1,728 |

The largest reversal of a hard-endpoint ordering was 0.007149 AP. More extreme calibrations on the other side still determined the minimax action. The largest amount by which any grid value exceeded its computed continuous maximum was below 2.3 × 10⁻¹⁶. Independent evaluation on binary weighted populations agreed with the maximizing witnesses to within 1.2 × 10⁻¹⁶ in 64 additional pair checks. These counts characterize this complete finite experiment. The ranking pairs share a support and carry no estimate of the frequency of reversals on deployed instruments.

## A counterexample at every severity cutoff

We derived this corollary after the registered test. Number six equal-mass strata by severity, with 1 most severe. Candidate A ranks them `2, 4, 5, 1, 6, 3`. Candidate B ranks them `6, 3, 2, 1, 4, 5`. A ties or beats B at each of the six possible hard severity cutoffs.

| Number of most severe strata counted positive | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AP of A | 1/4 | 3/4 | 2/3 | 41/48 | 29/30 | 1 |
| AP of B | 1/4 | 5/12 | 23/36 | 163/240 | 71/100 | 1 |

Give the six strata conditional positive probabilities $(1,t,t,0,0,0)$. This monotone calibration mixes the top-one and top-three definitions. The positive masses in score order, in units of one stratum, are $(t,0,0,1,0,t)$ for A and $(0,t,t,1,0,0)$ for B. Summing precision times positive mass at each score level gives

$$
\operatorname{AP}(B)-\operatorname{AP}(A)
= \frac{t(1-2t)}{12(1+2t)}.
$$

B wins for every $0<t<1/2$. At $t=1/5$, the two APs are $29/105$ and $17/60$, a reversal of $1/140$. Each stratum can be realized by five equally weighted records having 5, 1, 1, 0, 0 and 0 positives, respectively. This gives a directly supervised binary-label example. The witness is independent of the radial sensor and of the choice of dyadic budgets. Even a full set of binary-cutoff AP comparisons can give a false certificate about population AP under probabilistic labels.

The pair-product expansion of binary AP already appears in Aslam, Pavlu and Yilmaz. Literature checked after the experiment also gives a useful distinction. Robertson, Kanoulas and Yilmaz model a retrieval user who applies a common relevance threshold to documents. Here each stratum contributes its own positive and negative population mass to a single supervised PR curve, which produces cross-products between different cutoff profiles. We make no priority claim for the counterexample. [Aslam et al., A Sampling Technique for Efficiently Estimating Measures of Query Retrieval Performance](https://www.khoury.northeastern.edu/home/jaa/papers/AslamPaYi05.pdf) · [Robertson et al., Extending Average Precision to Graded Relevance Judgments, §2](https://www.khoury.northeastern.edu/home/ekanou/research/papers/mypapers/sigir10a.pdf)

## Scope failure and forward behavior

On 16 rings with eight directions per ring, two labelings retained all 128 rows, both score vectors, and prevalence 1/8. When the outer two rings were positive, the chosen radius ranking achieved AP 1 and the direction ranking achieved 0.125. Labeling one direction positive exchanged those APs, giving the identical selection 0.875 regret. Deployment needs a justified severity model. Under monotone risk the severity ordering itself is available to an operator permitted to deploy a new score. The candidate-selection problem matters when that operator must choose from an existing portfolio.

| Previously open source, both seeds | Full-risk and hard-cutoff minimax | Highest Spearman agreement with radius | Best AP for one top-5% radius cutoff |
| --- | --- | --- | --- |
| Wisconsin breast | KNN/5 | IForest/default | IForest/default |
| Cardio | KNN/25 | KNN/25 | KNN/25 |
| Ionosphere | KNN/5 | KNN/25 | KNN/5 |
| Pima | KNN/25 | KNN/25 | KNN/25 |

All eight forward calls returned a single candidate. The representative 2,000 × 64 batch with eight monitors took about 0.01 seconds for the selector inside a running Python process. Tied scores matched weighted supervised AP, and redundant feature columns gave a defined score on their observed linear span. Constant data and a saturated row space triggered abstention. Ordinary sample covariance can misorder severity under asymmetric contamination or infinite variance. No ADBench source was claimed to satisfy the severity model.

Run `python3 -m research.experiments.cohort_01.estimand.probe` from the repository root. The four-source public bank fingerprint is `6ab661ca92df7587328766829cfb3e085def8ab7ac5208df23eeaddaca4baff2`. Python 3.12.10, NumPy 2.4.4, SciPy 1.17.1 and scikit-learn 1.8.0 were used. The portable implementations remain research artifacts. [Full-risk optimizer](radial_ap_regret.py) · [Hard-cutoff control](radial_hard_cutoffs.py) · [Experiment and exact rational witness](probe.py)

| File | SHA-256 |
| --- | --- |
| `prediction.md` | `fe746ee3b33d7af56357160af168390474609ef94555259eeffc60307d266b9c` |
| `radial_ap_regret.py` | `a411e2f98ceab3b7d86d0e3c1b2bb9cb4081a01808477c0c22e04f1d198cef3c` |
| `radial_hard_cutoffs.py` | `c23603cef68a8022bcded9761b769280ca37fe8abd2e91a5cf8deed522f2a7ad` |
| `probe.py` | `8b8e5f0a272c4d66a398de6b6f2aeaefb1724d1c83052235ae2668a3f00131af` |

The unresolved mathematical question is which conditions bound the AP advantage of a stochastic calibration when all binary severity cutoffs favor the other ranking. An answer could separate immaterial reversals from ones large enough to justify changing a deployed decision.
