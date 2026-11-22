# A finite-sample AP certificate from two unlabeled cohorts

**Result, 2026-11-22.** For fixed binary detector scores, two unlabeled cohorts with known prevalence ordering and common class-conditional distributions can support a simultaneous confidence statement about which detector has higher population AP. In this three-candidate example the test certified one pair in every draw at ten million observations per cohort. At one million it certified that pair in 0.39% of draws. [Frozen experiment](binary_precision_prediction.md) · [Executable](finite_binary.py)

## Guarantee

Suppose each scoring rule has a prespecified alert set $E_i$ with $t_i=T(E_i)>0$. Let $d_i=H(E_i)-t_i$, and $q_i=d_i^2/t_i$. The binary-score special case of the population identity gives

$$
\operatorname{AP}_i=p(1+b\,d_i+b^2 q_i),\qquad b=(1-p)/(h-p)\ge1 .
$$

For two candidates, divide the AP difference by $p b$. The result is $\Delta(d+q)+(b-1)\Delta q$. Thus positive differences in both $d+q$ and $q$ suffice for strict AP dominance over every feasible mixing fraction. This sufficient condition can discard identifiable choices when dominance requires using an upper bound on $b$. [Population derivation](ordered_mixtures.md)

For $m$ fixed binary candidates, use two-sided Clopper–Pearson intervals with error $0.05/(2m)$ for each of the $2m$ alert probabilities. The union bound gives simultaneous coverage of at least 95% for all these parameters on one pair of independent iid samples. Within a sample, different detectors may flag the same rows. Propagating the covered intervals through subtraction, squaring and division gives valid bounds for each $q$ and $d+q$. Require both reported lower endpoints for one detector to exceed the other detector's upper endpoints. With probability at least 95%, **every** reported pairwise AP direction is correct under the shared-component model. A zero target lower endpoint gives an infinite upper bound on $q$. Data-dependent score selection or thresholds require independent evaluation samples or a different coverage argument. A repeated look at increasing sample sizes requires its own error spending. [Clopper and Pearson, 1934](https://doi.org/10.1093/biomet/26.4.404)

## Results

The two observable distributions, three binary candidates, seed and sample sizes were frozen before running the experiment. Each row summarizes 4,096 independent pairs of multinomial samples.

| Observations per cohort | Certified C above B | Abstained on C–B | Abstained on A–B | Abstained on A–C | Draws with any wrong direction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0 | 4,096 | 4,096 | 4,096 | 0 |
| 1,000,000 | 16 | 4,080 | 4,096 | 4,096 | 0 |
| 10,000,000 | 4,096 | 0 | 4,096 | 4,096 | 0 |

Wrong directions were counted over every admissible $b\in[1,30]$ and also over the stronger algebraic interval $b\ge1$. Both error counts were zero at each size. For the truly ambiguous A–B comparison either certificate would have been an error under some feasible world. The test also abstained on C–A, although C beats A over $[1,30]$. That comparison requires information the sufficient condition discards. The experiment reports power for this cell distribution. It supplies no estimate of power on incident logs, continuous scores or naturally occurring shifts.

**Decision.** The interval construction closes a finite-sample gap for some binary-score pairs. At the observed effect size its power was poor even at a million rows per cohort. The unresolved risk for deployment is within-class drift, which can reverse the population signal at any sample size. The one-batch transfer supplies a different information budget. [Exact drift example](ordered_mixtures.md)
