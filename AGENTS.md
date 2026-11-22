# Research work

For hypothesis generation, benchmark interpretation or a literature review, start with [the research memory](research/README.md). It links open questions, failed ideas, primary sources and the experiment protocol. Record the evidence and decision there when completing a research iteration.

The primary question is how closely a metric using only known-normal observations can reproduce the detector choice a labeled evaluation would make for a specified score, such as ROC AUC, PR AUC or F1. Fit candidates on normal data and select using normal data. After freezing the choice, compare its score on a hidden mixture of normal observations and real outliers with the best score in the same fitted portfolio. The first cohort studied a historical mixed-data setup. Record the track and supervised target with every result.
