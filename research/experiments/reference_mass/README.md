# When would a better mass estimate improve selection?

The deletion experiment showed that the same observations can support opposite target rankings. A complementary question for an existing metric is whether knowing its population value **exactly** would solve the decision in a stated regime. Here the supervised decision is precision at a fixed alert budget. Mass–volume methods measure how much reference volume a scoring function needs to cover a given fraction of observations. Uniform-reference anomaly ranking and the difficulty of estimating these curves in high dimensions are established work. [Clémençon and Robbiano](https://proceedings.mlr.press/v32/clemencon14.html) · [Goix](https://arxiv.org/abs/1607.01152)

Let detector `s` alert on a set `A_s` containing fraction `b` of the observed population. Let `π` be the anomaly prevalence and `P₁` the distribution of an anomalous row. Bayes' rule gives `precision_b(s) = π P₁(A_s) / b`. If a label-free reference `Q` really samples from `P₁`, the observable reference mass `Q(A_s)` orders detectors exactly at this budget. Unknown `π` cancels in comparisons. A uniform reference makes this the volume of the alert region, or equivalently one minus the normalized volume of its accepted complement. The identity concerns one operating point. Whole-curve AP gives credit across many budgets.

## Two finite populations with the same observations

Use three cells with volumes `(0.20, 0.05, 0.75)` and observed counts `(500, 500, 9,000)`. One detector alerts on cell A, another on B. Both flag precisely 5% of the batch. Keep the cells, detector scores, prevalence (5%) and uniform reference generator identical. Change only which mechanism generates the 500 anomalies.

| Anomaly counts in A, B, C | Precision of A | Precision of B | Choice using exact reference volume |
| --- | ---: | ---: | --- |
| 100, 25, 375 (proportional to volume) | 0.20 | 0.05 | A |
| 25, 100, 375 (more concentrated in B) | 0.05 | 0.20 | A |

Before the numerical check, the prediction is that increasing the number of uniform probes improves expected selected precision in the first world and worsens it in the second. The script builds both labeled worlds for the benchmark's supervised grader. It calculates the exact probability of choosing A from a finite number of uniform probes, allowing half credit when counts tie. Run it with `python -m research.experiments.reference_mass.probe`. [Executable example](probe.py)

This is a control for distinguishing estimation error from a mismatch between the reference and the anomaly mechanism. It examines a single mass–volume operating point, while the library's EM/MV baselines integrate curves over projections. It provides no performance estimate for those implementations on ADBench. For an extension, a domain could supply a range of credible fault generators. If each candidate alert set satisfies `|P₁(A_s) − Q(A_s)| ≤ ε`, a population reference-mass difference greater than `2ε` orders that pair correctly. An estimate from finite probes would also need to cover its sampling uncertainty. Whether such a bound is scientifically defensible for an application is itself a research question. A generator fitted using historical labels introduces historical supervision.

The calculation gave a probability of choosing A of 0.80209 with eight uniform probes and greater than 0.99999 with 512. Expected selected precision went from 0.17031 to 0.20000 in the proportional world and from 0.07969 to 0.05000 in the shifted world. Each number follows from the binomial sampling law and the benchmark's precision-at-budget grader.
