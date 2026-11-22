# Batch model-selection benchmark

The experiment asks how closely a metric using only **known-normal examples** can reproduce the detector choice a supervised evaluation would make. Name a target such as ROC AUC, PR AUC or F1 before grading. Compare the hidden test score of the chosen detector with the best test score among the same fitted candidates. This difference is the **selection gap** (regret). If the best ROC AUC is 0.94 and the chosen detector achieves 0.93, the gap is 0.01. The metric used to choose can have any numerical scale. The harness and its data live outside the public package.

## Normal-only protocol

1. A trusted preparer shuffles and samples up to `--max-samples` original rows without replacement (default 2,000). It reserves one quarter of each available class, rounded up, for the private test. This approximately preserves the sampled class mix. The other outliers are unused.
2. Of the remaining normal rows, two thirds (rounded down) fit the imputer, feature mask, scaler and detectors. The rest are separate normal examples for choosing a detector. These groups and the private group have disjoint row indices. The public bank contains the fitting normals, the selection normals and the fitted detectors' scores on the selection normals.
3. The selector freezes its ranking from that public bank. The private grader evaluates the same fitted detectors on the reserved mix of normal and outlier rows. The trusted preparer may cache these outcomes ahead of time. The researcher receives the permitted grade after submitting the choice. Public data contains neither private rows nor their scores or labels. A private sample with only one class is ungradable.

This is `prepare`'s default `--track normal-only`. The basic bank contains only real, normal observations. `--extra-views` is an explicit comparison condition for baselines that use normal-data refits and artificial Gaussian or uniform reference points. All those views are constructed solely from the fitting and selection normals. Record this option when interpreting a method that assumes a type of artificial anomaly. The old mixed-data experiment is available as `--track transductive`.

For agent-led research, start with the [research brief and handoff](../research/README.md). It includes a completed experiment with predictions recorded before grading and a counterexample for an existing mass–volume target.

## Try a development iteration

Run these commands from the repository root with Python 3.12. The first download uses the official ADBench repository. Subsequent runs reuse local files.

```sh
python -m pip install -e . -r benchmarks/requirements.txt

run=.bench-runs/normal-only
python -m benchmarks prepare --track normal-only --download --raw-dir .bench-runs/raw \
  --split dev --datasets 4_breastw 6_cardio 18_Ionosphere 29_Pima \
  --seeds 0 1 --out "$run/dev"

python -m benchmarks run --bank "$run/dev/public" \
  --ireos-rows 24 --out "$run/baselines"

python -m benchmarks run --bank "$run/dev/public" \
  --selector benchmarks/hypotheses/density_agreement.py \
  --out "$run/trial-001"

python -m benchmarks grade --truth "$run/dev/private" \
  --submissions "$run/baselines" "$run/trial-001" \
  --target auroc --reference iforest_default --out "$run/report-001"
```

For the next iteration, run the new selector against the **same public directory**. Grade both trial directories together with the cached baselines. Pass the previous selector's full name from its report as `--reference` for a paired comparison. Each run has a new output path and snapshots its source and hypothesis. Changing the portfolio, evidence, preprocessing or row sample creates a new bank. The grader checks bank IDs and requires every submitted selector to account for every task.

## Writing a hypothesis

A standalone Python module exports `select(batch)`. Return one value per candidate, with larger values preferred, or return `None` to abstain. The module docstring records the hypothesis. The CLI also accepts `--hypothesis`. A fresh Python process evaluates each custom selector on each task, with a default wall-clock budget of 120 seconds including startup. Set `--timeout` to change it. Each task gets read-only arrays.

| Input | Meaning |
| --- | --- |
| `batch.X` | Normal selection rows transformed using the fitting normals |
| `batch.scores` | Candidate × selection row. Larger values mean more anomalous |
| `batch.families` | One detector family per candidate, in matrix order |
| `batch.views` | `train` contains the transformed fitting normals. Optional additional views are described below. |

The included hypothesis gives each other detector family one vote on a candidate's top 5% of alerts. It subtracts the overlap expected if the two lists selected points independently within the same local-density quartiles. This tests whether agreement about *which rows* are unusual adds information beyond a shared preference for sparse regions. Fractional membership handles boundary ties. The example needs only the basic bank. [Example implementation](hypotheses/density_agreement.py)

For an ablation, replace the density bins with a single bin and rerun with a new output path. A useful experiment record contains the predicted effect, reference selector, bank ID, source hash, paired AP difference, failure count and the observation that motivates the next change. Development reports include per-task picks, oracle AP, sample and feature counts, and error reasons. These let an agent inspect a failed hypothesis without retraining the portfolio.

## Frozen protocol

- **Data.** The manifest lists all 47 ADBench Classical files and their Git blob hashes at one revision. It assigns 25 files to development, eight to validation and 14 to test. Related variants stay together, including the Cardiotocography, thyroid, Wisconsin breast and Landsat groups. The six datasets already used by the library's example notebooks are in development. Treat these source groups as an initial provenance audit. [Manifest](adbench.csv) · [Pinned ADBench collection](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/datasets/Classical)
- **Unit of evaluation.** One dataset and seed define one task. The `normal-only` track follows the three-way split above. The fixed 5% alert budget ranks even a completely normal selection set. It does not estimate the prevalence of anomalies in that set.
- **Candidate pool.** Two configurations each for Isolation Forest, KNN, LOF and HBOS. Isolation Forest uses its default sample count or 64, KNN and LOF use 5 or 25 neighbors, and HBOS uses 5 or 10 bins. Stochastic fits use the task seed. This small, family-balanced pool is intended for development. Freeze a wider pool and test sensitivity to duplicating or removing detector families before making a broad selection claim.
- **Cheap and expensive signals.** The basic bank supports all ten score-cluster indices, ASI/ASOI, RTM/EAG, Laplacian Score, SIREOS, MC/MCS/KNN centrality, UDR, HITS, ENS and both five-input Borda aggregations from the notebook. The fixed alert prior is 5% for partition and stability metrics. The adapters preserve each metric's native direction and apply score unification where needed. `--ireos-rows N` adds an explicitly named, shared-gamma IREOS comparison on the first N shuffled rows, capped at the batch size. It grades the resulting choice on the private target for the chosen track. The 24-row example is a smoke test.
- **Extra views.** NPD compares selection scores with scores on diagonal-Gaussian probes. Stability uses five 80% subsample refits scoring the same selection rows. EM/MV use 2,048 uniform reference points per projected unit box, with up to three five-feature projections. The box is defined by the available fitting and selection features. Arrays are `validation` and `gaussian` (candidate × row), `refits` (candidate × refit × row), and `projected_real` and `uniform` (projection × candidate × row). These methods report unavailable on a basic bank. On the normal-only track all real rows in these views are normal, and the main fitted detectors score the hidden test once for grading.
- **Reproducibility.** Inputs are checked against their pinned hashes. The public manifest records the environment versions, seeds, protocol and task-file hashes. Submissions contain source fingerprints, a custom selector's source snapshot, per-task ranks, status and timing. Reports include submission and grader fingerprints. Store the bank, inputs, environment and submissions used for a publication.

## Replaying the mixed-data pilot

The original four-source pilot and cohort 01 used `--track transductive`. In that setup all candidates and the selector saw the same unlabeled mixture of normal rows and outliers. The public scores were PyOD's native `decision_scores_` for the fitted rows, and the private outcomes described that same batch. Optional views used a further 70/30 split of the mixed data. These results do not measure selection from normal data alone.

To reproduce the old four-source quickstart, use the dataset and seed list above with `--track transductive --extra-views --out .bench-runs/dev`, then use `.bench-runs/dev/public` for the original baselines and trials. Historical bank files remain readable. For matching historical source fingerprints, the last pre-correction revision is `8b42e48`. Bank IDs guard against grading a normal-only submission with a transductive key.

## Reading a report

The `track` field identifies the preparation protocol. In a normal-only development report, `n_fit` and `n_select` count the normal rows for fitting and selection, and `n` counts the private test rows. Set `--target` to `auroc`, `pr_auc_trapezoid`, `ap`, `f1` or `precision_at_budget`. `ap` remains the command's default. `pr_auc_trapezoid` integrates the precision–recall curve by trapezoids. `ap` uses non-interpolated average precision, which ADBench calls `aucpr`. `precision_at_budget` evaluates the top 5% of test scores. [ADBench grader](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/myutils.py) · [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

**F1 needs an alert cutoff.** For each detector the normal-only preparer takes the empirical 95th percentile of its scores on the selection normals (inverse empirical CDF). A test score strictly above that fixed cutoff is an alert. This flags at most 5% of the selection normals, with ties at the cutoff treated as normal. The actual alert rate on new data can differ. The grader computes ordinary binary F1 from the test labels at that cutoff. F1 requires a newly prepared normal-only bank containing these cutoffs.

For each available supervised score the report gives the chosen detector's score and gap to the supervised best. `*_best_pick_rate` is the probability of picking one of the supervised winners under the tie policy. Within-task Spearman correlation measures agreement over the whole ranking where defined. A development report also names the actual `supervised_winners` on each task for the declared target. Validation and test expose aggregates only.

Tied winners receive the expected utility of a uniform choice among them. `uniform_model` is the mean utility of every candidate. Fixed controls expose the default Isolation Forest and every other individual configuration. Choose a strong fixed control on development and freeze it before validation. A failed, unavailable or abstaining selector falls back to default Isolation Forest. Status counts expose exactly how often it supplied a ranking. A single-class batch is ungradable and is omitted for every method, with its count reported.

The report averages seeds within datasets, datasets within source groups and source groups equally. Intervals for paired differences to the reference resample whole source groups 5,000 times. `source_wins` counts sources where the selector beats the reference on the declared target. A one-source run has no interval. Four-source smoke results can test the software and suggest a failure mode. A publication claim needs more independent sources and a frozen comparison.

## Giving agents access

The example commands run all roles locally for development. For a blind evaluation, a trusted preparer keeps the raw NPZs and the `private` directory in grader-only storage. Give the candidate execution environment a read-only copy of `public`, its method code, a writable submission destination and no network access. An external evaluator can call `grade` and return a report without executing candidate code. The surrounding execution environment supplies filesystem and network isolation.

Let agents query development reports freely. Freeze a shortlist before returning validation feedback, record each look, and run the final test once on a frozen revision. Validation and test reports expose aggregates. Repeated aggregate scores can still be used to overfit a held-out split. Public ADBench datasets can also be recognized. Reserve a provenance-disjoint external replication for a paper. [Blum and Hardt on adaptive leaderboards](https://proceedings.mlr.press/v37/blum15.html) · [MLE-bench's submission and grading design](https://github.com/openai/mle-bench)

An agent or researcher owns the hypothesis, decision rule, interpretation and next experiment. The package's formula tests and the benchmark's regression tests run with `python -m pytest`. Installing the benchmark requirements enables the PyOD integration test.

The initial audit records implementation edge cases, related methods and the result of the first ablation. [Research notes](RESEARCH_NOTES.md)
