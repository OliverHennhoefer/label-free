# Slides

[**Grading without an answer key**](slides/label-free-evaluation.pdf) (PDF, 20 slides)
is a visual introduction to label-free evaluation for people who know
unsupervised anomaly detection and supervised metrics such as ROC AUC and
average precision.

It gives each metric family in `labelfree` one picture and one metaphor:

| Metaphor | Metrics |
| --- | --- |
| The gap | RTM, EAG, score-cluster indices |
| The neighbourhood | ASI, ASOI, SIREOS, Laplacian Score |
| The fence | IREOS |
| The jury | MC, UDR, HITS, ENS |
| The retold story | Ranking stability |
| The tight net | Excess-Mass, Mass-Volume |
| The fire drill | NPD |

The second half covers the studies that tested these metrics against held-back
labels, led by Ma et al. (SIGKDD Explorations, 2023), and closes with
practical advice on using them anyway. The single-dataset chart comes from the
[metric comparison notebook](https://github.com/OliverHennhoefer/label-free/blob/main/examples/metric_comparison.ipynb).
