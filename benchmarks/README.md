# Batch model-selection benchmark

This repository-only harness measures what happens when an unlabeled criterion picks a PyOD detector and its hyperparameters for one batch. Its primary outcome is **AP regret**, the best candidate's average precision on that batch minus the selected candidate's average precision. A perfect choice has zero regret. The benchmark also measures paired gains over a fixed policy or another selector. The public package supplies metric functions. The harness supplies fixed experiments, evidence for the functions and a private grader.

For agent-led research, start with the [research brief and handoff](../research/README.md). It includes a completed experiment with predictions recorded before grading and a counterexample for an existing mass–volume target.

## Try a development iteration

Run these commands from the repository root with Python 3.12. The first download uses the official ADBench repository. Subsequent runs reuse local files.

```sh
python -m pip install -e . -r benchmarks/requirements.txt

python -m benchmarks prepare --download --raw-dir .bench-runs/raw \
  --split dev --datasets 4_breastw 6_cardio 18_Ionosphere 29_Pima \
  --seeds 0 1 --extra-views --out .bench-runs/dev

python -m benchmarks run --bank .bench-runs/dev/public \
  --ireos-rows 24 --out .bench-runs/baselines

python -m benchmarks run --bank .bench-runs/dev/public \
  --selector benchmarks/hypotheses/density_agreement.py \
  --out .bench-runs/trial-001

python -m benchmarks grade --truth .bench-runs/dev/private \
  --submissions .bench-runs/baselines .bench-runs/trial-001 \
  --reference iforest_default --out .bench-runs/report-001
```

For the next iteration, run the new selector against the **same public directory**. Grade both trial directories together with the cached baselines. Pass the previous selector's full name from its report as `--reference` for a paired comparison. Each run has a new output path and snapshots its source and hypothesis. Changing the portfolio, evidence, preprocessing or row sample creates a new bank. The grader checks bank IDs and requires every submitted selector to account for every task.

## Writing a hypothesis

A standalone Python module exports `select(batch)`. Return one value per candidate, with larger values preferred, or return `None` to abstain. The module docstring records the hypothesis. The CLI also accepts `--hypothesis`. A fresh Python process evaluates each custom selector on each task, with a default wall-clock budget of 120 seconds including startup. Set `--timeout` to change it. Each task gets read-only arrays.

| Input | Meaning |
| --- | --- |
| `batch.X` | Preprocessed rows in the batch |
| `batch.scores` | Candidate × row matrix. Larger values mean more anomalous |
| `batch.families` | One detector family per candidate, in matrix order |
| `batch.views` | Optional cached evidence described below |

The included hypothesis gives each other detector family one vote on a candidate's top 5% of alerts. It subtracts the overlap expected if the two lists selected points independently within the same local-density quartiles. This tests whether agreement about *which rows* are unusual adds information beyond a shared preference for sparse regions. Fractional membership handles boundary ties. The example needs only the basic bank. [Example implementation](hypotheses/density_agreement.py)

For an ablation, replace the density bins with a single bin and rerun with a new output path. A useful experiment record contains the predicted effect, reference selector, bank ID, source hash, paired AP difference, failure count and the observation that motivates the next change. Development reports include per-task picks, oracle AP, sample and feature counts, and error reasons. These let an agent inspect a failed hypothesis without retraining the portfolio.

## Frozen protocol

- **Data.** The manifest lists all 47 ADBench Classical files and their Git blob hashes at one revision. It assigns 25 files to development, eight to validation and 14 to test. Related variants stay together, including the Cardiotocography, thyroid, Wisconsin breast and Landsat groups. The six datasets already used by the library's example notebooks are in development. Treat these source groups as an initial provenance audit. [Manifest](adbench.csv) · [Pinned ADBench collection](https://github.com/Minqi824/ADBench/tree/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/datasets/Classical)
- **Unit of evaluation.** One dataset and seed define one transductive task. Sample up to 2,000 original rows without replacement using a label-blind permutation. Use `--max-samples` to set the cap. Fit median imputation, remove constant columns, standardize and fit each detector on the batch. Cache native PyOD `decision_scores_` in the same row order. Data preparation reads labels only for format validation and private grading. The code does not upsample small datasets or infer contamination from labels. Omit `--datasets` to prepare the entire requested split.
- **Candidate pool.** Two configurations each for Isolation Forest, KNN, LOF and HBOS. Isolation Forest uses its default sample count or 64, KNN and LOF use 5 or 25 neighbors, and HBOS uses 5 or 10 bins. Stochastic fits use the task seed. This small, family-balanced pool is intended for development. Freeze a wider pool and test sensitivity to duplicating or removing detector families before making a broad selection claim.
- **Cheap and expensive signals.** The basic bank supports all ten score-cluster indices, ASI/ASOI, RTM/EAG, Laplacian Score, SIREOS, MC/MCS/KNN centrality, UDR, HITS, ENS and both five-input Borda aggregations from the notebook. The fixed alert prior is 5% for partition and stability metrics. The adapters preserve each metric's native direction and apply score unification where needed. `--ireos-rows N` adds an explicitly named, shared-gamma IREOS comparison on the first N shuffled rows, capped at the batch size. It grades the resulting choice on the full batch. The 24-row example is a smoke test.
- **Extra views.** A shared label-blind 70/30 split supplies distinct training and evaluation rows. Imputation and standardization for this view fit on training rows. NPD compares held-out scores with scores on shared diagonal-Gaussian probes. Stability uses five 80% subsample refits on the training side, all scoring the same holdout. EM/MV use held-out scores and 2,048 shared uniform reference points per unit-box projection, with up to three five-feature projections and detector refits. The cached arrays are `validation` and `gaussian` (candidate × row), `refits` (candidate × refit × row), and `projected_real` and `uniform` (projection × candidate × row). Without these views the corresponding methods report unavailable. The primary supervised outcome always refers to the original batch fit.
- **Reproducibility.** Inputs are checked against their pinned hashes. The public manifest records the environment versions, seeds, protocol and task-file hashes. Submissions contain source fingerprints, a custom selector's source snapshot, per-task ranks, status and timing. Reports include submission and grader fingerprints. Store the bank, inputs, environment and submissions used for a publication.

## Reading a report

The grader privately computes AP using scikit-learn's non-interpolated definition, AUROC, trapezoidal PR area and precision at the fixed 5% alert budget. ADBench's implementation calls AP `aucpr`. The report preserves both PR definitions under distinct names. Every endpoint includes selected utility, regret to its own oracle, and within-task Spearman correlation where defined. [ADBench grader](https://github.com/Minqi824/ADBench/blob/3dac8221081e190f157d78e93bfa8867f90d0965/adbench/myutils.py) · [AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)

Tied winners receive the expected utility of a uniform choice among them. `uniform_model` is the mean utility of every candidate. Fixed controls expose the default Isolation Forest and every other individual configuration. Choose a strong fixed control on development and freeze it before validation. A failed, unavailable or abstaining selector falls back to default Isolation Forest. Status counts expose exactly how often it supplied a ranking. A single-class batch is ungradable and is omitted for every method, with its count reported.

The report averages seeds within datasets, datasets within source groups and source groups equally. Its interval for the paired AP difference resamples whole source groups 5,000 times. A one-source run has no interval. Four-source smoke results can test the software and suggest a failure mode. A publication claim needs more independent sources and a frozen comparison.

## Giving agents access

The example commands run all roles locally for development. For a blind evaluation, a trusted preparer keeps the raw NPZs and the `private` directory in grader-only storage. Give the candidate execution environment a read-only copy of `public`, its method code, a writable submission destination and no network access. An external evaluator can call `grade` and return a report without executing candidate code. The surrounding execution environment supplies filesystem and network isolation.

Let agents query development reports freely. Freeze a shortlist before returning validation feedback, record each look, and run the final test once on a frozen revision. Validation and test reports expose aggregates. Repeated aggregate scores can still be used to overfit a held-out split. Public ADBench datasets can also be recognized. Reserve a provenance-disjoint external replication for a paper. [Blum and Hardt on adaptive leaderboards](https://proceedings.mlr.press/v37/blum15.html) · [MLE-bench's submission and grading design](https://github.com/openai/mle-bench)

An agent or researcher owns the hypothesis, decision rule, interpretation and next experiment. The package's formula tests and the benchmark's regression tests run with `python -m pytest`. Installing the benchmark requirements enables the PyOD integration test.

The initial audit records implementation edge cases, related methods and the result of the first ablation. [Research notes](RESEARCH_NOTES.md)
