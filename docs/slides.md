# Slides

[**Evaluating anomaly detectors without labels**](slides/label-free-evaluation.pdf)
(PDF, 17 slides) is a short, visual introduction to internal evaluation for
people who know unsupervised anomaly detection and supervised metrics such as
ROC AUC and average precision.

Its premise is that the data at hand are normal; data that also contain
unlabeled anomalies are treated as a special case. It covers one metric family
per slide (level-set volume, synthetic anomalies, stability, consensus, score
separation, feature-space isolation, separability), each with a single figure, the idea behind it,
and when it fails. The second half summarizes the studies that tested these metrics
against held-back labels, led by Ma et al. (SIGKDD Explorations, 2023), and
closes with practical advice.

The deck's source is in [`docs/slides/src`](https://github.com/OliverHennhoefer/label-free/tree/main/docs/slides/src);
`build.py` there explains how to rebuild the PDF.
