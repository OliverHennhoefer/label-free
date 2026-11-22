# What the first research cohort found

**Decision, 2026-11-22.** The cohort produced two falsifiable single-batch proposals. One went through the six-source transfer and failed its comparison with a simpler control. The other failed the inventor's synthetic decision test and was withheld. The most promising theoretical result concerns identifying some AP orderings under a different, explicit information budget. The first cohort established no general ADBench improvement or publication priority. [Frozen decisions](submissions.md) · [Independent critique](critique/frozen_review.md)

## The fresh single-batch test

Six previously unopened development source groups, two seeds per group and eight candidate detectors. Each model choice was frozen before the evaluator released supervised results. One source gets one vote in the means. For abstentions the deployed policy uses default Isolation Forest. There were no ungradable tasks or runtime failures in the two reflection submissions. The four source groups from the original development bank had already informed hypothesis generation. [Allocation](allocation.json) · [Machine-readable decisions and outcomes for both banks](development.json)

| Selector | Original four sources, AP | Fresh six, AP | Fresh AP regret | Fresh AUROC | Fresh precision at 5% | Ranked on fresh tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Split reflection excess | 0.6126 | 0.1332 | 0.1629 | 0.6792 | 0.1127 | 3/12 |
| Matched clipped reflection | 0.6126 | 0.1654 | 0.1307 | 0.6927 | 0.1722 | 4/12 |
| Default Isolation Forest | 0.7047 | 0.1060 | 0.1902 | 0.6709 | 0.0650 | 12/12 |
| Fixed Isolation Forest, 64 samples | 0.7056 | 0.1160 | 0.1801 | 0.6697 | 0.0868 | 12/12 |
| S_Dbw | 0.7264 | 0.2062 | 0.0900 | 0.7579 | 0.2022 | 12/12 |
| HITS | 0.7180 | 0.1075 | 0.1886 | 0.6133 | 0.0825 | 12/12 |
| Pooled agreement | 0.7293 | 0.0973 | 0.1988 | 0.5958 | 0.0465 | 12/12 |
| Uniform detector choice | 0.6005 | 0.1509 | 0.1452 | 0.6715 | 0.1320 | 12/12 |

The split method won, tied and lost to the default on one, four and one of the six sources. Its mean paired advantage was +0.0272 AP (source-bootstrap 95% interval −0.0022 to +0.0838). Omitting Wine changes the mean to −0.0009. Against the clipped control the difference was −0.0322 (interval −0.0973 to +0.0005). S_Dbw had the highest mean AP on this transfer, the previously strongest pooled selector fell below random choice, and the mean best attainable AP in this portfolio was 0.2961. The absolute AP levels differ across banks and depend on their source mix and prevalence. The within-source comparisons answer the selection question.

| Fresh source | Split ranked / tasks | Split AP | Difference to default | Difference to clipped | Oracle headroom above fixed IForest/64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Letter | 0/2 | 0.0925 | 0 | 0 | 0.4728 |
| Mammography | 0/2 | 0.1496 | 0 | 0 | 0.0240 |
| Optdigits | 0/2 | 0.0467 | 0 | 0 | 0.1774 |
| Vertebral | 2/2 | 0.0888 | −0.0043 | +0.0011 | 0.0244 |
| Waveform | 0/2 | 0.0505 | 0 | 0 | 0.0731 |
| Wine | 1/2 | 0.3710 | +0.1676 | −0.1945 | 0.3090 |

## What changed in our understanding

- **The reflection center is scientific information.** If normal measurements have a known symmetry center and faults are one-sided within reflected pairs, probability mass left unmatched by reflection can reveal faults. The split estimator chose the correct ranking in all 24 paired synthetic replicates. On clean symmetric samples it estimated 9.89% contamination on average and abstained in only 3/24. A follow-up giving it the true center reduced the estimate to 0.14% and yielded 24/24 abstentions. Two physical centers can produce identical observations and opposite AP winners even with known prevalence and true nominal symmetry. Reopening this direction requires a center justified before deployment and a bound on symmetric faults. [Experiment and counterexamples](mechanism/result.md)
- **Bounding AP and improving a decision are different achievements.** For six rows, one ranking can tie or win under every hard severity cutoff and still lose in AP under a monotone mixture of those risks. The exact calculation found zero opposite strict minimax choices in 258,840 pairs of rankings on the registered support and calibration family. The richer model broke some ties. The registered test required a strict change in action with a meaningful regret gain. The arm preserved the counterexample and spent no fresh grade. [Derivation, witness and decision](estimand/result.md)
- **Another unlabeled cohort can change what is knowable.** If an operating protocol guarantees different ordered prevalences while the class-conditional distributions stay fixed, a detector's finite-score population AP difference has a sign determined by a linear function of one unknown parameter. Checking both feasible endpoints can prove a pairwise ranking. A conservative finite-sample construction for prespecified binary alert scores also controls the probability of any wrong reported pair at 5% for one look. In a difficult example it certified C above B in 16 of 4,096 draws at one million rows per cohort and in every draw at ten million. Drift within the normal class can reverse the signal. Ordinary random splits of one ADBench batch do not provide the required information. The nearest literature includes ordered-mixture AUROC and PU precision–recall correction. Publication priority of the AP comparison remains open. [Population proof and prior work](critique/ordered_mixtures.md) · [Finite-sample proof and power](critique/binary_precision_result.md)

The confidence statement in the last result matters. At one million rows per cohort, splitting an irrelevant cell into a rare marker of mass one in a million left every candidate's scores and true AP unchanged. Naively substituting empirical cell frequencies into the population bounds gave a false certificate in 1,632/4,096 draws on the fine partition and 376/4,096 on the original partition. This tests a direct plug-in implementation on the stated distribution. [Executable counterexample](check_sparse_support.py)

## Assessment at close

The loop demonstrated the ability to propose independent mechanisms, expose failure modes with controlled data and honor a decision to withhold a method. The code freeze before fresh grading made it possible to distinguish the Wine result from a prospectively defined domain advantage. One cohort gives no estimate of how often this process will discover publishable methods. The transferable scientific output is the set of identification conditions and counterexamples. A literature search and independent replication would still be needed for a novelty or practical performance claim.

For the original single-batch question, the clearest unresolved condition is access to independently calibrated symmetry and paired nuisance-only measurements. The center and fault semantics must be available before selection. The two-cohort result separately depends on a collection protocol that can justify ordered enrichment with stable normal and fault distributions. These ten ADBench source groups are now open development evidence. Validation and test each had zero looks. Six transferred sources give limited precision, the eight-model portfolio is narrow, and the access separation in this local agent run was procedural. This record closes the first cohort.

## Reproduction

Use the pinned benchmark dependencies. The original four-source bank and baseline commands are in the [benchmark quickstart](../../../benchmarks/README.md). The reflection probe both runs its synthetic experiment and materializes the byte-identical standalone clipped control. Choose fresh output directories for a replay. Run from the repository root.

```sh
python -m research.experiments.cohort_01.mechanism.probe --out .bench-runs/replay-cohort/control
python -m benchmarks prepare --download --raw-dir .bench-runs/replay-cohort/raw \
  --split dev --datasets 20_letter 23_mammography 26_optdigits 39_vertebral 41_Waveform 45_wine \
  --seeds 0 1 --extra-views --out .bench-runs/replay-cohort/bank

bank=.bench-runs/replay-cohort/bank/public
run=.bench-runs/replay-cohort
python -m benchmarks run --bank "$bank" --ireos-rows 24 --out "$run/baselines"
python -m benchmarks run --bank "$bank" --selector benchmarks/hypotheses/pooled_agreement.py --out "$run/pooled"
python -m benchmarks run --bank "$bank" --selector research/experiments/cohort_01/mechanism/reflection_excess.py --out "$run/split"
python -m benchmarks run --bank "$bank" --selector "$run/control/clipped_reflection.py" --out "$run/clipped"
python -m benchmarks grade --truth "$run/bank/private" --reference iforest_default \
  --submissions "$run/baselines" "$run/pooled" "$run/split" "$run/clipped" --out "$run/grade"
python -m research.experiments.cohort_01.compare "$run/grade/report.json" \
  --selectors reflection_excess@f19fa6c64a8b --references iforest_default clipped_reflection@a9e0ba400c81 s_dbw
```

Bank IDs start `6ab661ca92df` (original) and `209bfba0d0b8` (fresh). The retained evidence includes full bank, grader and report fingerprints. Run times contribute to report fingerprints and can differ on replay.
