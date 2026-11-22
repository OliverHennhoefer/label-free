"""Run the finite-population experiment, matched failure control and forward checks."""

import json
from fractions import Fraction
from itertools import permutations, product
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

from benchmarks.protocol import Batch, read_batch, top_membership

from . import radial_ap_regret as model
from . import radial_hard_cutoffs as rival


def batch_for(X, scores):
    return Batch(
        np.asarray(X), np.asarray(scores), tuple("monitor" for _ in scores), {}
    )


def crisp_ap(risk, score):
    # Each support point contributes two labeled atoms with known masses.
    y = np.tile([1, 0], len(risk))
    weights = np.column_stack((risk, 1 - risk)).ravel()
    return average_precision_score(y, np.repeat(score, 2), sample_weight=weights)


def strict_preference(margin):
    return (margin > 1e-10).astype(int) - (margin < -1e-10).astype(int)


def exact_audit(profiles, orders, scores, matrices, grid_ap, weights):
    # The three lowest-severity rows have identical profiles on this support.
    # Relabeling them preserves the full AP function, reducing the cost of an
    # exact audit of the complete six-point experiment.
    _, representatives, multiplicities = np.unique(
        profiles[orders].reshape(len(orders), -1),
        axis=0,
        return_index=True,
        return_counts=True,
    )
    prevalence = profiles.mean(axis=0)
    count_both_tied = int(np.sum(multiplicities * (multiplicities - 1) // 2))
    counts = {
        "strict_choice_switches": 0,
        "hard_ties": count_both_tied,
        "continuous_ties": count_both_tied,
        "hard_ties_broken_in_continuous": 0,
        "hard_weak_dominance_with_some_strict_endpoint_reversed": 0,
    }
    best_strict_improvement = 0.0
    largest_grid_excess = 0.0
    largest_violation = None
    for position, i in enumerate(representatives):
        for later in range(position + 1, len(representatives)):
            j = representatives[later]
            repetitions = int(multiplicities[position] * multiplicities[later])
            difference = matrices[i] - matrices[j]
            hard = np.diag(difference) / prevalence
            hard_i = max(0.0, -hard.min())
            hard_j = max(0.0, hard.max())
            gain_i, wi = model.maximum_gain(difference, prevalence)
            gain_j, wj = model.maximum_gain(-difference, prevalence)
            exact_i, exact_j = max(0, gain_j), max(0, gain_i)
            grid = grid_ap[i] - grid_ap[j]
            largest_grid_excess = max(
                largest_grid_excess, grid.max() - gain_i, -grid.min() - gain_j
            )
            hard_pick = strict_preference(np.array(hard_j - hard_i))
            full_pick = strict_preference(np.array(exact_j - exact_i))
            if hard_pick == 0:
                counts["hard_ties"] += repetitions
                if full_pick != 0:
                    counts["hard_ties_broken_in_continuous"] += repetitions
            if full_pick == 0:
                counts["continuous_ties"] += repetitions
            if hard_pick * full_pick < 0:
                counts["strict_choice_switches"] += repetitions
                best_strict_improvement = max(
                    best_strict_improvement, abs(exact_i - exact_j)
                )

            if hard.min() >= -1e-10 and hard.max() > 1e-10 and gain_j > 1e-10:
                dominant, challenger, gain, witness, rational = (
                    i,
                    j,
                    gain_j,
                    wj,
                    weights[np.argmin(grid)],
                )
            elif hard.max() <= 1e-10 and hard.min() < -1e-10 and gain_i > 1e-10:
                dominant, challenger, gain, witness, rational = (
                    j,
                    i,
                    gain_i,
                    wi,
                    weights[np.argmax(grid)],
                )
            else:
                continue
            counts["hard_weak_dominance_with_some_strict_endpoint_reversed"] += (
                repetitions
            )
            if (
                largest_violation is None
                or gain > largest_violation["largest_reversal"] + 1e-12
            ):
                chosen_scores = scores[[dominant, challenger]]
                q = profiles @ rational
                binary_y = np.concatenate(
                    [np.arange(20) < int(np.rint(20 * rate)) for rate in q]
                )
                supervised = [
                    average_precision_score(binary_y, np.repeat(s, 20))
                    for s in chosen_scores
                ]
                assert np.allclose(supervised, [crisp_ap(q, s) for s in chosen_scores])
                largest_violation = {
                    "largest_reversal": float(gain),
                    "orders_dominant_then_challenger": orders[
                        [dominant, challenger]
                    ].tolist(),
                    "aps_at_vertices": [
                        [crisp_ap(p, s) for p in profiles.T] for s in chosen_scores
                    ],
                    "exact_weights_in_increasing_tail_mass": witness.tolist(),
                    "grid_weights": rational.tolist(),
                    "replicated_supervised_AP_at_grid_weights": supervised,
                    "direct_population_AP_at_exact_weights": [
                        crisp_ap(profiles @ witness, s) for s in chosen_scores
                    ],
                }
    assert largest_grid_excess < 1e-9
    return {
        **counts,
        "ranking_types": len(representatives),
        "pairs_of_different_types": len(representatives)
        * (len(representatives) - 1)
        // 2,
        "largest_grid_excess_over_exact_all_pairs": float(largest_grid_excess),
        "largest_regret_improvement_strict_switch": float(best_strict_improvement),
        "weak_dominance_witness": largest_violation,
    }


def six_points():
    X = np.array([-3, -2, -1, 1, 2, 4], dtype=float)[:, None]
    profiles = model.radial_profiles(X)
    prevalence = profiles.mean(axis=0)
    orders = np.array(list(permutations(range(len(X)))))
    scores = -np.argsort(orders, axis=1)
    # A grid reference evaluated directly from population precision and recall.
    denominator = 20
    compositions = [
        (*prefix, denominator - sum(prefix))
        for prefix in product(range(denominator + 1), repeat=len(prevalence) - 1)
        if sum(prefix) <= denominator
    ]
    weights = np.asarray(compositions) / denominator
    risks = weights @ profiles.T
    grid_ap = []
    for order in orders:
        ordered = risks[:, order]
        precision = np.cumsum(ordered, axis=1) / np.arange(1, len(X) + 1)
        grid_ap.append(np.sum(ordered * precision, axis=1) / ordered.sum(axis=1))
    grid_ap = np.asarray(grid_ap)
    vertices = np.max(weights, axis=1) == 1
    hard_ap = grid_ap[:, vertices]
    counts = {
        "pairs": 0,
        "hard_ties": 0,
        "grid_ties": 0,
        "opposite_strict_choices": 0,
        "hard_dominance_reversals": 0,
    }
    strongest = None
    for i in range(len(orders) - 1):
        delta = grid_ap[i] - grid_ap[i + 1 :]
        lower = delta.min(axis=1)
        upper = delta.max(axis=1)
        grid_margin = np.maximum(upper, 0) - np.maximum(-lower, 0)
        delta_hard = hard_ap[i] - hard_ap[i + 1 :]
        lower_hard = delta_hard.min(axis=1)
        upper_hard = delta_hard.max(axis=1)
        hard_margin = np.maximum(upper_hard, 0) - np.maximum(-lower_hard, 0)
        grid_pick = strict_preference(grid_margin)
        hard_pick = strict_preference(hard_margin)
        switches = grid_pick * hard_pick < 0
        counts["pairs"] += len(delta)
        counts["grid_ties"] += int(np.count_nonzero(grid_pick == 0))
        counts["hard_ties"] += int(np.count_nonzero(hard_pick == 0))
        counts["opposite_strict_choices"] += int(np.count_nonzero(switches))
        reversals = ((lower_hard >= -1e-10) & (lower < -1e-10)) | (
            (upper_hard <= 1e-10) & (upper > 1e-10)
        )
        counts["hard_dominance_reversals"] += int(np.count_nonzero(reversals))
        if switches.any():
            local = np.argmax(np.where(switches, np.abs(grid_margin), -1))
            best = abs(grid_margin[local])
            if strongest is None or best > strongest[0] + 1e-12:
                strongest = (float(best), i, i + 1 + local)

    matrices = np.array([model.ap_matrix(score, profiles) for score in scores])
    rng = np.random.default_rng(405601)
    # An exact maximum must cover every grid point and attain its value at its
    # witness when independently evaluated on a binary, weighted population.
    largest_grid_excess = 0.0
    largest_witness_error = 0.0
    for i, j in rng.choice(len(orders), size=(64, 2)):
        gain, witness = model.maximum_gain(matrices[i] - matrices[j], prevalence)
        largest_grid_excess = max(
            largest_grid_excess, float((grid_ap[i] - grid_ap[j]).max() - gain)
        )
        risk = profiles @ witness
        observed = crisp_ap(risk, scores[i]) - crisp_ap(risk, scores[j])
        largest_witness_error = max(largest_witness_error, abs(float(observed - gain)))
    assert largest_grid_excess < 1e-9 and largest_witness_error < 1e-9
    result = {
        **counts,
        "grid_profiles": len(weights),
        "largest_grid_excess_64_pairs": largest_grid_excess,
        "largest_attainment_error_64_pairs": largest_witness_error,
        "exact_audit": exact_audit(
            profiles, orders, scores, matrices, grid_ap, weights
        ),
    }
    if strongest is None:
        return result

    grid_improvement, i, j = strongest
    batch = batch_for(X, scores[[i, j]])
    exact_regrets = -model.select(batch)
    hard_regrets = -rival.select(batch)
    hard_choice = int(np.argmin(hard_regrets))
    soft_choice = int(np.argmin(exact_regrets))
    vertex_aps = [
        [crisp_ap(profile, score) for profile in profiles.T] for score in batch.scores
    ]
    delta = grid_ap[i] - grid_ap[j]
    grid_worst = int(np.argmin(delta) if hard_choice == 0 else np.argmax(delta))
    risk = risks[grid_worst]
    counts_positive = np.rint(risk * denominator).astype(int)
    y = np.concatenate([np.arange(denominator) < count for count in counts_positive])
    replicated_ap = [
        average_precision_score(y, np.repeat(score, denominator))
        for score in batch.scores
    ]
    assert np.allclose(replicated_ap, grid_ap[[i, j], grid_worst])
    assert exact_regrets[0] + 1e-10 >= max(0, -delta.min())
    assert exact_regrets[1] + 1e-10 >= max(0, delta.max())
    exact_witness = model.maximum_gain(
        matrices[[i, j][1 - hard_choice]] - matrices[[i, j][hard_choice]], prevalence
    )[1]
    result["strongest_grid_pair"] = {
        "grid_improvement": grid_improvement,
        "descending_orders_in_zero_based_X": orders[[i, j]].tolist(),
        "hard_regrets": hard_regrets.tolist(),
        "soft_regrets": exact_regrets.tolist(),
        "hard_choice": hard_choice,
        "soft_choice": soft_choice,
        "soft_improvement": float(
            exact_regrets[hard_choice] - exact_regrets[soft_choice]
        ),
        "vertex_ap_in_increasing_tail_mass": vertex_aps,
        "tail_masses": prevalence.tolist(),
        "worst_grid_weights_for_hard_choice": weights[grid_worst].tolist(),
        "exact_worst_weights_for_hard_choice": exact_witness.tolist(),
        "supervised_AP_at_grid_weights": replicated_ap,
    }
    return result


def all_cutoff_example():
    """Algebraic corollary after the registered test, on its same six severities."""
    orders = [[2, 4, 5, 1, 6, 3], [6, 3, 2, 1, 4, 5]]

    def exact_ap(risk, order):
        ranked = [risk[i - 1] for i in order]
        return sum(
            q * sum(ranked[:k]) / k for k, q in enumerate(ranked, start=1)
        ) / sum(ranked)

    hard = [
        [
            exact_ap([Fraction(i <= cutoff) for i in range(1, 7)], order)
            for cutoff in range(1, 7)
        ]
        for order in orders
    ]
    assert all(a >= b for a, b in zip(*hard, strict=True))
    t = Fraction(1, 5)
    risk = [Fraction(1), t, t, Fraction(0), Fraction(0), Fraction(0)]
    soft = [exact_ap(risk, order) for order in orders]
    assert soft == [Fraction(29, 105), Fraction(17, 60)] and soft[1] - soft[
        0
    ] == Fraction(1, 140)
    return {
        "orders_by_severity": orders,
        "all_cutoff_AP": [[str(v) for v in a] for a in hard],
        "stochastic_AP_at_one_fifth": [str(v) for v in soft],
    }


def ring_control():
    ring = np.repeat(np.arange(1, 17), 8)
    direction = np.tile(np.arange(8), 16)
    theta = direction * np.pi / 4
    X = np.column_stack((np.cos(theta), np.sin(theta))) * ring[:, None]
    scores = np.stack((ring, (direction == 0).astype(float)))
    batch = batch_for(X, scores)
    chosen = int(np.argmax(model.select(batch)))
    outcome = {"choice": chosen, "hard_choice": int(np.argmax(rival.select(batch)))}
    for label, y in (("outer", ring >= 15), ("direction", direction == 0)):
        values = np.array([average_precision_score(y, s) for s in scores])
        outcome[label] = {
            "AP": values.tolist(),
            "regret": float(values.max() - values[chosen]),
        }
    return outcome


def practical():
    rng = np.random.default_rng(13091)
    X = rng.normal(size=(2000, 64))
    scores = np.stack([X[:, :end].var(axis=1) for end in range(8, 65, 8)])
    batch = batch_for(X, scores)
    start = perf_counter()
    values = model.select(batch)
    elapsed = perf_counter() - start
    assert values.shape == (8,) and np.isfinite(values).all()
    assert model.select(batch_for(np.ones((12, 3)), np.ones((2, 12)))) is None
    assert model.select(batch_for(np.eye(12), np.ones((2, 12)))) is None
    # Verify tied score levels, singular columns, row-order invariance and the
    # agreement of the hard control with the vertices of the main model.
    X = np.array([[-3], [-2], [-1], [1], [2], [4]], dtype=float)
    score = np.array([[1, 2, 2, 1, 3, 3], [2, 2, 1, 1, 1, 3]])
    batch = batch_for(np.hstack([X, 2 * X]), score)
    profiles = model.radial_profiles(batch.X)
    ws = np.array([0.13, 0.36, 0.4, 0.11])
    risk = profiles @ ws
    aps = np.array(
        [ws @ model.ap_matrix(s, profiles) @ ws / risk.mean() for s in score]
    )
    assert np.allclose(aps, [crisp_ap(risk, s) for s in score])
    vertex_aps = np.array([[crisp_ap(p, s) for p in profiles.T] for s in score])
    assert np.allclose(
        -rival.select(batch), (vertex_aps.max(axis=0) - vertex_aps).max(axis=1)
    )
    perm = rng.permutation(len(X))
    assert np.allclose(
        model.select(batch), model.select(batch_for(batch.X[perm], score[:, perm]))
    )
    return {
        "seconds_2000x64_eight_scores": elapsed,
        "tied_support_weighted_supervised_AP": aps.tolist(),
    }


def public_choices():
    root = Path(".bench-runs/dev/public")
    manifest = json.loads((root / "bank.json").read_text())
    candidates = np.array(manifest["candidates"])
    result = {}
    for task, checksum in manifest["tasks"].items():
        batch = read_batch(root, task, checksum, manifest["families"])
        start = perf_counter()
        robust = model.select(batch)
        seconds = perf_counter() - start
        hard = rival.select(batch)
        if robust is None:
            result[task] = {"abstained": True, "seconds": seconds}
            continue
        centered = batch.X - batch.X.mean(axis=0)
        left, singular, _ = np.linalg.svd(centered, full_matrices=False)
        rank = np.count_nonzero(
            singular > singular[0] * max(centered.shape) * np.finfo(float).eps
        )
        radius = np.round(np.sum(left[:, :rank] ** 2, axis=1), 12)
        ranks = rankdata(radius)
        agreement = np.array(
            [np.corrcoef(ranks, rankdata(s))[0, 1] for s in batch.scores]
        )
        single = top_membership(radius, int(np.ceil(0.05 * len(radius))))
        fixed = np.array([crisp_ap(single, s) for s in batch.scores])

        def picks(v):
            return candidates[np.isclose(v, v.max(), rtol=0, atol=1e-12)].tolist()

        result[task] = {
            "robust": picks(robust),
            "hard": picks(hard),
            "spearman": picks(agreement),
            "single_5pct": picks(fixed),
            "seconds": seconds,
        }
    return result


if __name__ == "__main__":
    outcome = {
        "six_points": six_points(),
        "all_cutoff_example": all_cutoff_example(),
        "ring_control": ring_control(),
        "practical": practical(),
        "open_public_forward_only": public_choices(),
    }
    print(json.dumps(outcome, indent=2, allow_nan=False))
