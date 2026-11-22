"""Run the preregistered synthetic reflection controls and freeze the matched rival."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score

from benchmarks.protocol import top_membership
from research.experiments.cohort_01.mechanism.reflection_excess import (
    anomaly_weights,
    average_precisions,
)

HERE = Path(__file__).parent
VERTICES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def construct_pair(seed):
    rng = np.random.default_rng(seed)
    jitter = rng.integers(-2, 3, size=(4, 60, 3))
    clouds = [radius * VERTICES[:, None, :] + jitter for radius in (30, 80)]
    half_core = rng.integers(-6, 7, size=(620, 3))
    core = np.concatenate([half_core, -half_core])
    permutation = rng.permutation(2000)
    scores = np.zeros((2, 2000))
    scores[0, :240] = 1
    scores[1, 240:480] = 1
    scores = scores[:, permutation]
    for active in (0, 1):
        inactive = 1 - active
        features = np.concatenate(
            [
                *(cloud.reshape(-1, 3) for cloud in clouds),
                core,
                -clouds[active][:, :10].reshape(-1, 3),
                -clouds[inactive].reshape(-1, 3),
            ]
        )
        excess = np.zeros(2000, dtype=bool)
        active_alerts = slice(240 * active, 240 * (active + 1))
        excess[active_alerts] = np.tile(np.arange(60) >= 10, 4)
        central_faults = excess.copy()
        central_faults[480 : 480 + 240] = True
        central_faults[480 + 620 : 480 + 620 + 240] = True
        bilateral_faults = excess.copy()
        bilateral_faults[240 * inactive : 240 * (inactive + 1)] = True
        bilateral_faults[-240:] = True
        labels = [
            truth[permutation] for truth in (excess, central_faults, bilateral_faults)
        ]
        yield features[permutation], scores, labels


def iid_null(seed):
    rng = np.random.default_rng(5000 + seed)
    group = rng.choice(5, 2000, p=[0.52, 0.12, 0.12, 0.12, 0.12])
    centers = np.array([0, 30, -30, 80, -80])[group, None]
    features = centers * VERTICES[rng.integers(4, size=2000)] + rng.integers(
        -2, 3, size=(2000, 3)
    )
    core = group == 0
    features[core] = rng.integers(-6, 7, size=(core.sum(), 3))
    return features


def radius_weights(features):
    count = len(features)
    k = int(np.ceil(np.sqrt(count)))
    radius = np.empty(count)
    for start in range(0, count, 128):
        end = min(start + 128, count)
        distances = cdist(features[start:end], features, metric="sqeuclidean")
        distances[np.arange(end - start), np.arange(start, end)] = np.inf
        distances.partition(k - 1, axis=1)
        radius[start:end] = distances[:, k - 1]
    return top_membership(radius, int(np.ceil(0.05 * count)))


def admitted(weights):
    return bool(2 / np.sqrt(len(weights)) < weights.mean() <= 0.5)


def credit(values, active):
    if np.isclose(values[0], values[1], rtol=0, atol=1e-12):
        return 0.5
    return float(np.argmax(values) == active)


def symmetric_sample(features):
    values, counts = np.unique(features, axis=0, return_counts=True)
    return bool(
        np.array_equal(values, -values[::-1]) and np.array_equal(counts, counts[::-1])
    )


def measurements(features, scores):
    record = {}
    for name, clipped in (("split", False), ("clipped", True)):
        weights = anomaly_weights(features, clipped=clipped)
        record[name] = {
            "prevalence": float(weights.mean()),
            "admitted": admitted(weights),
            "ap": average_precisions(scores, weights).tolist(),
        }
    record["radius_ap"] = average_precisions(scores, radius_weights(features)).tolist()
    return record


def run():
    worlds, nulls = [], []
    for seed in range(24):
        first_scores = None
        for active, (features, scores, labelings) in enumerate(construct_pair(seed)):
            if first_scores is None:
                first_scores = scores.copy()
            assert np.array_equal(first_scores, scores)
            assert all(symmetric_sample(features[~labels]) for labels in labelings)
            truth_ap = [
                [
                    float(average_precision_score(labels, candidate))
                    for candidate in scores
                ]
                for labels in labelings
            ]
            record = measurements(features, scores)
            record.update(
                seed=seed,
                active=active,
                label_prevalence=[float(labels.mean()) for labels in labelings],
                truth_ap=truth_ap,
            )
            worlds.append(record)
        null = {"seed": seed}
        features = iid_null(seed)
        for name, clipped in (("split", False), ("clipped", True)):
            weights = anomaly_weights(features, clipped=clipped)
            null[name] = {
                "prevalence": float(weights.mean()),
                "admitted": admitted(weights),
            }
        nulls.append(null)
    return worlds, nulls


def summarize(worlds, nulls):
    summary = {}
    for name in ("split", "clipped"):
        wins = [
            w[name]["admitted"] and credit(w[name]["ap"], w["active"]) == 1
            for w in worlds
        ]
        ap_error = [np.abs(np.array(w[name]["ap"]) - w["truth_ap"][0]) for w in worlds]
        summary[name] = {
            "paired_both_right_of_24": int(np.reshape(wins, (-1, 2)).all(axis=1).sum()),
            "world_admitted_of_48": sum(w[name]["admitted"] for w in worlds),
            "mean_world_prevalence": float(
                np.mean([w[name]["prevalence"] for w in worlds])
            ),
            "world_prevalence_mae": float(
                np.mean([abs(w[name]["prevalence"] - 0.10) for w in worlds])
            ),
            "active_ap_mae": float(
                np.mean([e[w["active"]] for w, e in zip(worlds, ap_error)])
            ),
            "inactive_ap_mae": float(
                np.mean([e[1 - w["active"]] for w, e in zip(worlds, ap_error)])
            ),
            "null_mean_prevalence": float(
                np.mean([r[name]["prevalence"] for r in nulls])
            ),
            "null_prevalence_mae": float(
                np.mean([abs(r[name]["prevalence"]) for r in nulls])
            ),
            "null_abstained_of_24": sum(not r[name]["admitted"] for r in nulls),
        }
    summary["radius_mean_win_credit"] = float(
        np.mean([credit(w["radius_ap"], w["active"]) for w in worlds])
    )
    summary["same_prevalence_reversals_of_48"] = sum(
        credit(w["truth_ap"][1], w["active"]) == 1
        and credit(w["truth_ap"][2], w["active"]) == 0
        and w["label_prevalence"][1] == w["label_prevalence"][2] == 0.34
        for w in worlds
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    source = HERE / "reflection_excess.py"
    control = args.out / "clipped_reflection.py"
    old = "return rank_batch(batch, clipped=False)"
    new = "return rank_batch(batch, clipped=True)"
    code = source.read_text()
    assert code.count(old) == 1
    control.write_text(code.replace(old, new))
    worlds, nulls = run()
    summary = summarize(worlds, nulls)
    record = {
        "sha256": {
            p.name: sha256(p)
            for p in (HERE / "prediction.md", source, control, Path(__file__))
        },
        "summary": summary,
        "worlds": worlds,
        "nulls": nulls,
    }
    (args.out / "probe.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
