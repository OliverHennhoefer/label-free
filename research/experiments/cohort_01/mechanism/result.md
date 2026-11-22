# Reflected excess found the right shell and often the wrong anomaly rate

**Decision.** Retire the median-centered rule for general batch selection. It found the AP-leading detector after a controlled change in the data geometry. It failed both the independent symmetric-null screen and the fresh-source comparison with its clipped rival. The population calculation survives for measurements with an externally established nominal symmetry and a bound on the faults that can be hidden on both sides of it. [Prospective design](prediction.md) · [Frozen selector](reflection_excess.py)

## Which observation changed the choice?

Two detectors each flagged 12% of the rows. Their alert scores were byte-identical across two populations. The populations differed in which alert group had few observations at its reflected coordinates. Abundant reflected companions identified rare, valid observations in one group. The missing companions in the other accounted for 10% of the population. The active detector's AP was 0.8333 and the other's was 0.1000. We checked symmetry of the empirical nominal distribution at every individual feature value for all three label assignments in every world. [Executable experiment](probe.py)

| Preregistered check | Split orientation and measurement | Clipped on the same sample |
| --- | ---: | ---: |
| Both choices correct, out of 24 paired populations | 24 | 24 |
| Populations receiving a ranking, out of 48 | 48 | 48 |
| Mean estimated prevalence, true value 0.10 | 0.1079 | 0.2087 |
| Mean absolute prevalence error | 0.0361 | 0.1087 |
| Mean absolute AP error for the active detector | 0.2852 | 0.3183 |
| Mean absolute AP error for the inactive detector | 0.0310 | 0.1041 |
| Mean absolute estimated prevalence, true null value 0 | 0.0998 | 0.1843 |
| Abstentions on 24 iid symmetric nulls | 3 | 0 |

A local-rarity selector gave the two detectors equal credit, 0.50 per world. The matched alert clusters had the same translated local geometry and the same number of observations. Any deterministic score-only policy would return the same choice before and after the companion swap. Both reflection estimators exploited the change. Splitting the estimation did not improve the synthetic choices. In 16/48 worlds its signed AP estimate for the active detector exceeded one, reaching 1.748. A correct ordering was compatible with large errors in the reported utility.

The prevalence-error criterion of 0.04 passed narrowly in the contaminated populations and failed on the symmetric nulls. The registered null screen also required at least 20 abstentions. A post-hoc synthetic diagnostic held the simulated observations, folds and partition settings fixed, then supplied the true physical zero. The median had missed zero in 45/72 partition-training folds. Supplying zero changed the mean signed null estimate from 0.09895 to 0.00139 and its mean absolute value from 0.09976 to 0.01261. All 24 nulls then met the abstention threshold. Charging signed mass even to empty favored cells changed the frozen null mean by less than 0.00001. This diagnostic localizes the practical error on this generator. It gives no deployment procedure for finding the true center. [Diagnostic](null_diagnostic.py)

## Two identification limits

**A known center with faults on both sides.** Write the observed density as p and its reflected density as pᵣ. Symmetry of the nominal subdensity implies that every compatible positive subdensity is e + b, where e = max(p − pᵣ, 0), b is nonnegative and symmetric, and e + b ≤ p. Conversely, every such choice has a symmetric nominal remainder. Thus the observed data fix the excess e and constrain the symmetric mass b only by availability. The maximal symmetric-background formula is already present in Arias-Castro and Jiang. The decision calculation below translates the unresolved component into AP regret. [Arias-Castro and Jiang, Theorem 1](https://arxiv.org/html/2106.13925)

In the preregistered failure case, start with the 0.10 identified excess and add 0.24 symmetric fault mass. Putting it in the center gives AP 0.4851 to the active detector and 0.3400 to its rival. Putting it on both sides of the valid shell gives AP 0.4851 and 0.5729. All features, detector scores, nominal symmetry and the true anomaly prevalence of 0.34 are common to these two labelings. The AP winner reversed in 48/48 constructed worlds.

A post-hoc derivation gives the precise amount of hidden mass this particular decision can tolerate. For a binary ranking whose alert group has mass h and positive mass m, with total prevalence π, the step AP is π − m + m²/(πh). Here h = 0.12 for both candidates. If the total hidden symmetric mass is b, the active alerts retain at least 0.10 positive mass. At most b/2 can enter the rival alerts because their reflected companions lie outside those alerts. For b up to 0.24, the sum of the two positive alert masses is at least 0.10, greater than πh. Subtracting the two AP expressions then gives the same sign as the difference in their positive alert masses. Every compatible allocation preserves the active winner for b < 0.20. Allocating b equally across the two sides of the rival shell achieves a tie at b = 0.20 and a reversal for 0.20 < b ≤ 0.24. This threshold applies to the equal-budget binary scores in this constructed population. [Numerical checks](identification.py)

**An unknown center, even with fully one-sided faults.** Let the observations be uniform on [0,1] and let the known anomaly rate be ε, strictly between 0 and one half. One compatible sensor has nominal observations uniform on [ε,1] and faults on [0,ε]. Its nominal center is (1+ε)/2. Another has nominal observations uniform on [0,1−ε] and faults on [1−ε,1], with nominal center (1−ε)/2. In each case the normal distribution is symmetric and the reflected fault interval lies outside the observed support except for an endpoint. A detector alerting on the left ε fraction and one alerting on the right ε fraction have APs (1, ε) in the first model and (ε, 1) in the second. The complete observation distribution and the true rate are identical. A policy choosing the left detector with probability q has respective regrets (1−q)(1−ε) and q(1−ε). Its worst-case regret is at least (1−ε)/2, attained at q = 1/2. For ε = 0.05 the bound is 0.475 AP. The argument works for arbitrarily small positive ε, even with infinitely many observations. This deduction followed the failed null test. The executable checks it on finite uniform grids at ε = 0.001, 0.05 and 0.25. [Numerical checks](identification.py)

## Fresh development transfer

The referee graded both frozen rules after the joint submission freeze at commit `42ee77d`. The fresh bank SHA-256 begins `209bfba0d0b8`. Six sources had two detector seeds apiece. Abstentions receive the default Isolation Forest's AP. Split reflection ranked 3/12 tasks and abstained on nine. Clipped reflection ranked 4/12. Each returned a unique winner whenever it ranked. Source-weighted mean AP was 0.13317 for split and 0.16542 for clipped. Their respective AP regrets were 0.16294 and 0.13070. Default Isolation Forest scored 0.10596, IForest/64 scored 0.11600, HITS scored 0.10753, S_Dbw scored 0.20616, pooled agreement scored 0.09728 and uniform detector choice scored 0.15089. Split lost 0.03224 AP to the matched rival. Its gain of 0.02722 over the default had a 95% source-bootstrap interval of [−0.00215, +0.08380] and became −0.00086 when wine was omitted. [Cohort protocol and referee record](../README.md)

| Fresh source | Split AP | Clipped AP | Split minus default | Split minus clipped |
| --- | ---: | ---: | ---: | ---: |
| Letter | 0.09245 | 0.09245 | 0 | 0 |
| Mammography | 0.14963 | 0.14963 | 0 | 0 |
| Optdigits | 0.04665 | 0.04665 | 0 | 0 |
| Vertebral | 0.08883 | 0.08778 | −0.00430 | +0.00105 |
| Waveform | 0.05053 | 0.05053 | 0 | 0 |
| Wine | 0.37095 | 0.56546 | +0.16759 | −0.19451 |

Both rules abstained twice on each of the four sources with identical results. On Vertebral, split selected KNN/25 and clipped selected HBOS/10 on seed 0. Both picked IForest/64 on seed 1. On Wine, clipped selected HBOS/5 both times. Split withheld that choice on seed 0 and received the default's utility. On the four previously opened sources the two rules each scored 0.61257, against 0.70466 for the default. Selecting HBOS/5 on both Ionosphere seeds cost 0.37965 AP there. They gained 0.01126 on Pima and equaled the default on Cardio and Wisconsin breast. No ADBench source had prospectively established nominal reflection symmetry. The frozen method fails the registered transfer decision.

## Provenance and next question

The statistical ingredients have direct precedents. Arias-Castro and Jiang identify the largest symmetric background in a mixture. Xiang and colleagues estimate the ratio of reflected and observed densities for a scalar local false discovery target. Tombs and Lester discriminate real and transformed observations to test physical symmetries, including effects of measurement filters. The experiment studied attribution of positive mass to candidate score tails and the resulting AP decision. Priority for these selection-specific calculations remains open. [Arias-Castro and Jiang](https://arxiv.org/abs/2106.13925) · [Xiang et al.](https://arxiv.org/abs/2606.11548) · [Tombs and Lester](https://arxiv.org/abs/2111.05442)

The next discriminating question is whether a sensor with a separately calibrated physical zero and a bound on symmetric faults can certify one candidate's advantage under real measurement drift. The construction above gives a motivation for collecting those facts independently of the test batch. The ADBench grades provide no such calibration.

Reproduce the synthetic experiment and the two follow-up checks from the repository root, choosing a fresh output directory.

```sh
python3 -m research.experiments.cohort_01.mechanism.probe --out .bench-runs/reflection-reproduction
python3 -m research.experiments.cohort_01.mechanism.null_diagnostic
python3 -m research.experiments.cohort_01.mechanism.identification
```

The probe emits its matched standalone comparator by changing only the final value of `clipped` in the selector and saves all per-seed outputs. The prediction, split selector, clipped control, probe source and original JSON result have SHA-256 prefixes `1f2dda85aab2`, `f19fa6c64a8b`, `a9e0ba400c81`, `f9c0f2c73e18` and `c80dd16147fd`. A 2,000 × 64 Gaussian timing check took 0.013 seconds for weight estimation on this machine. Its mean signed weight was zero. Global sign reversal preserved the weights, monotone score transformation preserved AP and constant feature input produced an abstention.
