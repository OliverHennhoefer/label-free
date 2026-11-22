"""Verify the two population counterexamples on exact finite grids."""

import json

import numpy as np
from sklearn.metrics import average_precision_score


def unknown_baseline(rows, positives):
    x = np.arange(rows)
    left = x < positives
    right = x >= rows - positives
    labels = [left, right]
    # The integer grid keeps reflections and support comparisons exact.
    twice_centers = [rows - 1 + positives, rows - 1 - positives]
    for truth, twice_center in zip(labels, twice_centers, strict=True):
        assert np.array_equal(x[~truth], twice_center - x[~truth][::-1])
        assert not np.intersect1d(twice_center - x[truth], x[truth]).size
    utilities = [
        [float(average_precision_score(truth, detector)) for detector in labels]
        for truth in labels
    ]
    prevalence = positives / rows
    assert np.allclose(utilities, [[1, prevalence], [prevalence, 1]])
    return {
        "prevalence": prevalence,
        "ap": utilities,
        "minimax_regret": (1 - prevalence) / 2,
    }


def binary_ap(prevalence, positive_alert_mass, alert_mass):
    recall = positive_alert_mass / prevalence
    precision = positive_alert_mass / alert_mass
    return recall * precision + (1 - recall) * prevalence


def hidden_symmetric_mass():
    examples = []
    for hidden in (0, 0.18, 0.20, 0.22, 0.24):
        prevalence = 0.10 + hidden
        active = binary_ap(prevalence, 0.10, 0.12)
        rival = binary_ap(prevalence, hidden / 2, 0.12)
        examples.append({"hidden": hidden, "active_ap": active, "rival_ap": rival})
    return examples


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "unknown_baseline": [unknown_baseline(2000, k) for k in (2, 100, 500)],
                "hidden_symmetric_mass": hidden_symmetric_mass(),
            },
            indent=2,
        )
    )
