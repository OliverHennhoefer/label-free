# Experiment 01 · What did deletion tell us?

**Decision.** Retire this version for general batch selection. Its proposed effect was present in the constructed population. A simpler rival explained the same synthetic choices. On four previously open ADBench sources, the incremental mean AP from refitting was 0.00037, and the selected detectors lost 0.02057 AP to default Isolation Forest. The preregistered screen stopped here. Validation and test remained sealed. [Predictions and stopping decision](design.md) · [Selector and matched rival](../../../benchmarks/hypotheses/conditional_repair.py)

## Where the prediction held

The unit of replication was one simulated population. In 20/20 seeds the relation-violating ranking had positive repair relative to random deletion and beat the ranking that found valid extremes. The direct-residual rival preferred it in 20/20 as well. When relation violations were labeled anomalous, the mean APs of the two rankings were 1.000 and 0.027. With identical observed rows and scores, labeling the extreme operating regime anomalous changed those APs to 0.047 and 1.000. The oracle reversed in every seed. This establishes a limit on what either selector can infer and demonstrates the sign of the repair effect in this construction. Both observations can hold simultaneously.

## Where the methods made different decisions

Four development sources, two detector seeds per source, eight PyOD configurations. Results average seeds within source and give sources equal weight. The two methods produced untied choices on all eight tasks, agreed on five, and each differed from the default frequently. The fixed configuration shown below had the strongest mean AP among fixed detectors on these same open sources.

| Selector | Selected AP | AP regret | AUROC | Trapezoidal PR area | Precision at 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conditional repair | 0.6841 | 0.0798 | 0.8515 | 0.6826 | 0.7633 |
| Direct residual | 0.6837 | 0.0802 | 0.8540 | 0.6822 | 0.7490 |
| Default Isolation Forest | 0.7047 | 0.0593 | 0.8573 | 0.7033 | 0.7905 |
| Fixed Isolation Forest, 64 samples | 0.7056 | 0.0583 | 0.8554 | 0.7042 | 0.7944 |
| HITS | 0.7180 | 0.0460 | 0.8719 | 0.7165 | 0.7915 |
| S_Dbw | 0.7264 | 0.0376 | 0.8756 | 0.7250 | 0.7878 |
| Pooled agreement | 0.7293 | 0.0347 | 0.8819 | 0.7278 | 0.7801 |
| Uniform detector choice | 0.6005 | 0.1635 | 0.7754 | 0.5986 | 0.6363 |

| Source | Repair AP | Repair minus default AP | Repair minus direct-residual AP |
| --- | ---: | ---: | ---: |
| Wisconsin breast (`4_breastw`) | 0.9837 | +0.0159 | +0.0407 |
| Cardio | 0.3831 | −0.1794 | 0.0000 |
| Ionosphere | 0.8493 | +0.0565 | −0.0392 |
| Pima | 0.5203 | +0.0247 | 0.0000 |

The effect of the refit almost cancels across sources. On the Wisconsin data it selected IForest/64 instead of KNN/25 on both seeds. On Ionosphere seed 0 it selected KNN/25 while the direct rival picked KNN/5. On Cardio both preferred KNN/25 and both incurred the same loss. The source-bootstrap interval for the mean AP difference to default was [−0.1284, +0.0463] across just four sources. These data do not settle whether repair helps a prospectively defined population of relation-breaking faults.

**Handoff.** The toy demonstration cannot distinguish refitting from simply detecting high residuals. Reopening this idea would require a setting justified independently of these grades where a group has *modest individual residuals* but materially changes a fitted relationship. Predict different choices for the two selectors there, including a case where refitting is harmful. For a different line, the follow-up asks what assumptions make an existing reference-volume quantity identify a supervised decision. [Reference-mass example](../reference_mass/README.md)

## Reproduce from the repository root

Create the four-source bank and run its built-in baselines and `trial-001` as in the [benchmark quickstart](../../../benchmarks/README.md). The pooled control is the exact source from the earlier one-bin ablation. Run its command below once if the saved `trial-002` directory is absent. Choose fresh output directories when repeating an experiment.

```sh
python -m benchmarks run --bank .bench-runs/dev/public \
  --selector benchmarks/hypotheses/pooled_agreement.py --out .bench-runs/trial-002

python -m research.experiments.conditional_repair.probe --out .bench-runs/repair-v1/probe
python -m benchmarks run --bank .bench-runs/dev/public \
  --selector .bench-runs/repair-v1/probe/conditional_repair.py --out .bench-runs/repair-v1/repair
python -m benchmarks run --bank .bench-runs/dev/public \
  --selector .bench-runs/repair-v1/probe/direct_residual.py --out .bench-runs/repair-v1/residual

python -m benchmarks grade --truth .bench-runs/dev/private \
  --submissions .bench-runs/baselines .bench-runs/trial-001 .bench-runs/trial-002 \
  .bench-runs/repair-v1/repair .bench-runs/repair-v1/residual \
  --reference iforest_default --out .bench-runs/repair-v1/grade
```

The original public bank ID starts `6ab661ca92df`. SHA-256 prefixes for the prediction document, repair selector and standalone direct rival are `72fffb906458`, `2d273d73bc33` and `532bf7785748`. The synthetic `probe.json` has raw results for every seed and full hashes. Each custom `submission.json` contains the frozen rankings, and `grade/report.json` contains all per-source picks and endpoints on this development bank.
