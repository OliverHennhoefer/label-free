# What evidence can distinguish two anomaly definitions?

**Independent referee note, 2026-11-22.** Written before seeing either proposed selector or the six-source bank. The search covered internal outlier model selection, unlabeled evaluation of ranked classifiers, proxy anomalies, and ranking under ordered class proportions.

## A control with identical observations

Consider two recorded events in the same batch, each occurring in a disjoint 5% of rows. Detector A puts the first event above all other rows, and detector B does the same for the second. In one deployment the first event is a fault and the second is a scheduled test. In another deployment the maintenance policy reverses their meanings. Both have 5% faults. Every feature, score, refit, generated probe and pairwise agreement can be reused byte for byte.

| Operational definition | A's AP | B's AP |
| --- | ---: | ---: |
| First event requires intervention | 1.00 | 0.05 |
| Second event requires intervention | 0.05 | 1.00 |

Both scores have two levels. Under the adverse definition all positives enter at the final threshold, where precision equals prevalence. Every deterministic choice incurs 0.95 regret in one world. Even random selection incurs at least 0.475 expected regret in the worse world. The bound concerns unrestricted semantics on this input. A documented upstream distinction between a fault and a scheduled test can invalidate one of the two worlds. Tan and Clarke previously formalized the related idea of maximizing effectiveness differences between unlabeled rankings. [Exact calculation](exact_controls.py) · [Tan and Clarke, 2015](https://arxiv.org/abs/1408.3587)

## Where to look for a rival

- A selector that infers detector skill from rank agreement has a close mathematical comparator in SUMMA. Its conditional-independence model relates the covariance of ranks to AUROC. Correlated detector families and a shared nuisance can induce agreement even when the designated faults occur elsewhere. A useful control retains the nuisance and changes which event requires intervention. Idan's anomaly-specific work additionally ranks candidates by agreement and diversity. [Ahsen, Vogel and Stolovitzky, JMLR 2019](https://jmlr.org/papers/v20/18-094.html) · [Jaffe et al., AISTATS 2016](https://proceedings.mlr.press/v51/jaffe16.html) · [Idan, ECAI 2024](https://arxiv.org/abs/2410.14579)
- For synthetic anomalies, the nearest comparison depends on what the generator knows. AutoUAD scores real and Gaussian data. AutoAnoEval uses feature semantics, an LLM and a tree on 20 tabular tasks. An externally specified operating condition conveys information that could distinguish the two worlds above. Its contribution can be tested by comparing the same detector and generator with the condition exchanged. [Dai and Fan, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bf375a8dabbae40def018957ea68842a-Abstract-Conference.html) · [Yoon et al., EACL 2026](https://aclanthology.org/2026.findings-eacl.183/)
- For claims about choosing a local or global detector, Bouman et al. found different winners in two groups of datasets defined by supervised performance profiles. That established heterogeneity. An observable gate that assigns a fresh batch to the right group adds a separate empirical claim. [Bouman, Bukhsh and Heskes, JMLR 2024](https://jmlr.org/papers/v25/23-0570.html)

## Reading the transfer

Six source groups give six paired differences. A second seed reuses the source and may contain the same rows under another permutation when the sample is below the cap. Compare the gain with the four-source fixed winner and the closest mechanism-specific rival. For each source show the available improvement above the fixed detector, gain or harm, and when a fallback supplied the decision. A leave-one-source-out mean reveals whether one source drives the result. Ma et al. previously found fixed Isolation Forest competitive against internal selectors across 39 tasks. [Cohort protocol](../README.md) · [Ma et al., empirical comparison](https://www.andrew.cmu.edu/user/lakoglu/pubs/23-SigKDDExplor-UnsupervisedModelSelection-WhitePaperSurvey.pdf)

Even under an independent, equiprobable-sign null with no ties, five wins in six gives a two-sided exact sign-test probability of 14/64. Six wins gives 2/64. A source bootstrap cannot create new domains. Six-source agreement with a predeclared prediction can justify a lead. Independent replication needs fresh source provenance and a comparator frozen before feedback. MacrOData provides candidate semantic, one-versus-rest and synthetic collections. Its source relationships and the proposed mechanism would determine the replication units. [Ding et al., KDD 2026](https://arxiv.org/abs/2602.09329)

**Decision.** Keep the paired-definition control available for the frozen submissions. Two unlabeled cohorts with a previously documented prevalence ordering offer a separate route to identifying supervised choices. The finite-population result below establishes the precise scope and gives a failure under within-class drift. [Ordered-cohort calculation](ordered_mixtures.md)
