# Research memory

Start here when resuming the project. The question is when evidence available on an unlabeled batch can identify a useful detector for a specified supervised decision. The current experiments use ADBench and PyOD. The research record also has room for other populations, including time series.

## What we know so far

| Finding or open question | Scope and evidence | Decision |
| --- | --- | --- |
| AP and AUROC can choose different detectors. A fixed detector can be competitive. | On both open Pima seeds, HBOS/10 led on AP and KNN/25 on AUROC. On the four open sources, default Isolation Forest averaged 0.705 AP. [Development evidence](AGENT_BRIEF.md) | Name the supervised decision before judging a proposal. |
| Density-conditioning a family vote failed its first ablation. | Mean AP 0.720 with quartile conditioning and 0.729 with a single bin, four development sources. [Audit](../benchmarks/RESEARCH_NOTES.md) · [Frozen single-bin comparator](../benchmarks/hypotheses/pooled_agreement.py) | Retain the simpler version as a comparator. |
| Deleting flagged rows can improve typical feature predictions. The incremental selection benefit is unresolved. | The effect held in 20/20 constructed populations. On the four open sources, repair and direct residual scoring achieved 0.6841 and 0.6837 AP. [Prediction](experiments/conditional_repair/design.md) · [Outcome and harms](experiments/conditional_repair/result.md) | Retire this specification for general selection. A future setting would have to discriminate between deletion and individual residuals. |
| At a fixed alert budget a correct anomaly reference orders detectors by precision. A wrong reference can make more accurate estimation harmful. | Exact calculation in two finite populations. It concerns one operating point. [Derivation and executable](experiments/reference_mass/README.md) · [Relevant literature](literature/2026-11-21-label-free-selection.md) | Require independent justification for a proposed reference distribution. |

The [researcher brief](AGENT_BRIEF.md) can be handed to an agent as written. The [benchmark](../benchmarks/README.md) defines inputs, outputs, splits and grading. The [library audit](../benchmarks/RESEARCH_NOTES.md) remains the record of implementation caveats. The first cohort is tracked in its [experiment directory](experiments/cohort_01/README.md).

## Where to write the next piece of evidence

- **Experiment.** Use `research/experiments/<question>/`. Save assumptions, a rival and a discriminating prediction before new grades. Add a separate result with the compared choices, bank and code identifiers, important failures, decision and next open question. Keep a prior prediction intact when an idea changes. Link a follow-up or a correction. Small executable probes can live next to their records. Reusable benchmark selectors go in `benchmarks/hypotheses/`.
- **Literature review.** Use `research/literature/YYYY-MM-DD-<question>.md` when a question calls for a search. Record the question, date, search scope, primary sources, what each actually establishes, and what the review changes. Link a later review when new evidence changes a claim. A citation can belong to several experiments without duplicating the review.
- **Handoff.** Read this index and the linked evidence for the open question. End an iteration by adding the decision, its scope and a link here. Keep negative results and competing explanations searchable. Use `rg -ni 'residual|volume' research` to find earlier work by concept. Parallel researchers can write separate experiment notes and have the evaluator reconcile the index after grading.
- **Run data.** `.bench-runs/` is a local cache for score banks, source snapshots, submissions and machine reports. The durable record links the bank and selector fingerprints, commands and interpretable result. For a paper, archive the exact allowed artifacts in access-controlled, immutable storage and record their checksums. Keep raw labels, grading keys and restricted feedback with the evaluator. Record which sources have become development evidence so future agents know which observations are independent.
