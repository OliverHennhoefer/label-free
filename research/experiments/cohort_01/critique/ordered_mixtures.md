# Which detector choices can two unlabeled cohorts identify?

**Finite-population derivation, 2026-11-22.** Consider a deployment cohort $T$ and an additional unlabeled cohort $H$. A sampling or operating protocol establishes that the anomaly prevalences obey $0<p<h\leq 1$. Within each class the joint feature distribution is common to both cohorts. Write $N$ for normal observations, $A$ for anomalies and $D=H-T$ for the signed difference of the observed probability measures. All candidate scoring functions, preprocessing and score thresholds are fixed across cohorts. The empirical benchmark has one target batch and supplies neither an ordered second cohort nor this sampling guarantee.

## Fixed alert budget

Write $T=(1-p)N+pA$ and $H=(1-h)N+hA$. Then $A=T+bD$ where $b=(1-p)/(h-p)>0$. For a candidate's alert set $E$ with deployment mass $T(E)=q$, its deployment precision is

$$
\operatorname{Precision}_T(E)=\frac{pA(E)}q=p+\frac{pb}{q}D(E).
$$

Across candidates, larger $D(E)$ therefore means higher precision at that budget. Each candidate sets its own threshold in $T$ and applies that same threshold in $H$. Randomizing boundary ties with the same acceptance probability in both cohorts achieves equal alert mass in $T$. Recomputing the quantile in $H$ sets both masses to $q$ and erases the signal.

## A sharp AP comparison on finite support

Let the observable joint feature space have finitely many atoms of positive deployment mass $t_i$. Put $d_i=H_i-t_i$ and assume $D\ne0$. Allow all model-compatible prevalences. The possible values of $b$ are exactly

$$
1\le b\le b_{\max}=\min_{i:d_i<0}\frac{t_i}{-d_i}.
$$

Necessity follows from $h\le1$ and $A_i=t_i+b d_i\ge0$. For sufficiency choose $a>0$ small enough that $N=T-aD$ is nonnegative. Then $p=a/(a+b)$ and $h=(a+1)/(a+b)$ produce both original laws with these $A,N$. If both cohorts must be contaminated by normals, omit $b=1$ and use its limit. If $H=T$ and $h>p$, the common-component model forces $A=N$ and all scores have AP $p$.

For one candidate, sort its distinct score levels in descending order. Let $t_g,d_g$ denote each level's deployment and difference masses, with cumulative values $T_g,D_g$. Step-function AP handles score ties by adding all rows at each level. The recall increment is $t_g+b d_g$, and precision at that point is $p(1+bD_g/T_g)$. Multiplying and summing gives

$$
\operatorname{AP}=p(1+bL+b^2Q),\qquad
L=\sum_g t_g D_g/T_g,\qquad Q=\sum_g d_g D_g/T_g.
$$

For two candidates the sign of their AP difference equals the sign of $\Delta L+b\,\Delta Q$. **Checking two endpoints decides whether one candidate weakly dominates throughout the identified set.** Opposite endpoint signs mean both winning orders are compatible with the observed cohorts. The argument is about population distributions and non-interpolated AP. High-dimensional empirical frequency tables often have disjoint observed support even where the true distributions overlap. Any finite-sample decision would need prespecified score partitions and simultaneous uncertainty control for their masses. A coarser population partition enlarges the feasible interval and can sacrifice power.

### Two different AP winners with the same deployment prevalence

Let $T=(0.05,0.50,0.45)$ and $H=(0.055,0.510,0.435)$. Candidate A flags cell one, B flags cell two, and C flags either. Each assigns one score to flagged rows and another to the rest. Here $b_{\max}=30$. A and B exchange AP order at $b=50/3$. C beats both over the entire feasible interval. With a 5% budget and fractional boundary ties, A's excess alert mass is 0.005, B's is 0.001 and C's is about 0.00136. A therefore has the highest precision at 5% in every admissible world.

| Target prevalence | Enriched prevalence | $b$ | A AP | B AP | C AP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.1900 | 10 | 0.11000 | 0.11200 | 0.11909 |
| 0.10 | 0.1375 | 24 | 0.14080 | 0.13552 | 0.15956 |

The two rows share the entire observable distribution of both cohorts and the true deployment prevalence. Both rows admit legitimate normal and anomalous class distributions. Knowing the deployed class prevalence still leaves the A–B AP choice unresolved. The two-endpoint comparison can nevertheless identify C as the AP winner. The executable checks both decompositions and computes AP directly with exact fractions. [Exact controls](exact_controls.py)

The evaluator independently checked the algebra in 256 finite populations using NumPy `default_rng(117)`, 3–11 cells, independent Dirichlet$(1,\ldots,1)$ draws for $N,A$, $p$ uniform on $(0.03,0.35)$, $h$ uniform on $(p+0.02,0.95)$, and four candidate scores with integer levels 0 through 4. Weighted scikit-learn AP used a normal and an anomalous copy of each cell with masses $(1-p)N_i$ and $pA_i$. Its maximum absolute discrepancy from the polynomial was $1.7\times10^{-15}$. Of 1,536 pairwise comparisons, 1,439 had endpoint margins of the same sign with magnitude above $10^{-10}$, and all 1,439 matched the oracle order at the generating mixture. These checks use exact population masses as inputs.

## How the operational assumption can fail

Take $T=(0.1,0.1,0.8)$, $p=0.1$, and $A=(0.4,0.1,0.5)$. The two detectors alert on cells one and two respectively. Their deployment precisions are 0.40 and 0.10 at the same 10% budget. Keep $A$ identical across cohorts and double the prevalence to $h=0.2$, but change the normal distribution in $H$ to $(0.02,0.20,0.78)$. The two observable alert-mass changes are now $-0.004$ and $+0.080$. The selector prefers the detector that loses 0.30 precision. Known enrichment alone therefore leaves serious ambiguity. An intervention that changes sensor readings within normal units, or that changes the mix of anomaly subtypes, can cause this failure.

## Prior work and decision

Xie et al. establish how ordered unlabeled mixtures support AUROC optimization. Jain et al. recover precision–recall and ROC curves with class proportions in positive–unlabeled and noisy-positive settings. Katz-Samuels et al. study conditions that identify class distributions from mixtures. The derivation here asks which **finite-score AP orderings** their underlying mixture geometry can identify with unknown proportions. The search located these close results and leaves priority of the endpoint test open. [Xie, Liu and Li, AAAI 2024](https://arxiv.org/abs/2305.15776) · [Jain, White and Radivojac, AAAI 2017](https://arxiv.org/abs/1702.00518) · [Katz-Samuels, Blanchard and Scott, JMLR 2019](https://jmlr.org/papers/v20/17-576.html)

**Decision.** Preserve the result as a separate research direction for batches collected under justified prevalence enrichment. A prospective test needs an operationally defensible common class-conditional law, a nuisance-shift control, and separate evaluation samples for a common fitted detector. A random split of one batch has no prevalence-order guarantee.
