"""Adapters from the published library functions to one value per candidate."""

from time import perf_counter

import numpy as np

import labelfree as lf

from .protocol import TAIL, ranking


def assess(function, size):
    start = perf_counter()
    try:
        values = function()
        if values is None:
            record = {"status": "abstained"}
        else:
            record = {"status": "ok", "ranks": ranking(values, size).tolist()}
    except Exception as error:  # noqa: BLE001 -- a failing hypothesis is a measured result
        record = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
    return {**record, "seconds": perf_counter() - start}


def baselines(batch, ireos_rows):
    X, S, views = batch.X, batch.scores, batch.views
    count = len(S)
    results = {}

    def add(name, function):
        results[name] = assess(function, count)

    def each(name, function, direction=1):
        add(name, lambda: [direction * function(s) for s in S])

    # Compute the ten indices together, then orient each independently.
    start = perf_counter()
    try:
        indices = [lf.score_cluster_metrics(s, contamination=TAIL) for s in S]
    except (ValueError, ArithmeticError, RuntimeError) as error:
        for name in lf.SCORE_CLUSTER_INDICES:
            results[name] = {"status": "failed", "reason": str(error)}
    else:
        for name, direction in lf.SCORE_CLUSTER_INDICES.items():
            sign = 1 if direction == "higher" else -1
            add(
                name,
                lambda sign=sign, name=name: [sign * item[name] for item in indices],
            )
            results[name]["seconds"] = perf_counter() - start

    each("rtm", lf.relative_top_median_score)
    each("eag", lf.expected_anomaly_gap_score)
    each("asi", lambda s: lf.asi_score(X, s, contamination=TAIL))
    each("asoi", lambda s: lf.asoi_score(X, s, contamination=TAIL))
    each("laplacian", lambda s: lf.laplacian_score(X, s), -1)
    each("sireos", lambda s: lf.sireos_score(X, lf.unify_scores(s)), -1)
    add("mc", lambda: lf.model_centrality_scores(S))
    add(
        "mcs",
        lambda: lf.model_centrality_scores(
            S, n_peers=round(np.sqrt(count)), random_state=0
        ),
    )
    add("knn_centrality", lambda: lf.model_centrality_scores(S, n_neighbors=2))
    add("udr", lambda: lf.udr_scores(S, batch.families))
    add("hits", lambda: lf.hits_scores(S))
    add("ens", lambda: lf.ens_scores(S))

    if "refits" in views:
        add(
            "npd",
            lambda: [
                lf.normalized_pseudo_discrepancy_score(a, b)
                for a, b in zip(views["validation"], views["gaussian"], strict=True)
            ],
        )
        add(
            "stability",
            lambda: [
                lf.ranking_stability_score(runs, contamination=TAIL)
                for runs in views["refits"]
            ],
        )

        def volumes(function):
            return np.mean(
                [
                    [
                        function(real, uniform, support_volume=1.0)
                        for real, uniform in zip(a, b, strict=True)
                    ]
                    for a, b in zip(
                        views["projected_real"], views["uniform"], strict=True
                    )
                ],
                axis=0,
            )

        add("em", lambda: volumes(lf.excess_mass_auc))
        add("mv", lambda: -volumes(lf.mass_volume_auc))
    else:
        for name in ("npd", "stability", "em", "mv"):
            results[name] = {
                "status": "unavailable",
                "reason": "Prepare with --extra-views",
            }
    if ireos_rows is not None:
        # Rows are already shuffled without labels. All candidates use the same
        # rows, one unification convention and one shared automatic gamma range.
        def ireos():
            P = np.stack([lf.unify_scores(s[:ireos_rows]) for s in S])
            return lf.ireos_scores(X[:ireos_rows], P)

        add(f"ireos@{ireos_rows}", ireos)

    def aggregate(function):
        names = ("asoi", "laplacian", "sireos", "mc", "hits")
        if any(results[name]["status"] != "ok" for name in names):
            raise ValueError("A component of the five-metric notebook ensemble failed")
        return function(
            [results[name]["ranks"] for name in names], higher_is_better=False
        )

    add("borda", lambda: aggregate(lf.borda_scores))
    add("robust_borda", lambda: aggregate(lf.robust_borda_scores))
    return results
