"""Post-hoc synthetic diagnosis of center error and empty measurement cells."""

import json

import numpy as np

from research.experiments.cohort_01.mechanism.probe import iid_null
from research.experiments.cohort_01.mechanism.reflection_excess import (
    _assign,
    _fit_prototypes,
)


def diagnose(features, *, known_center):
    n = len(features)
    folds = np.array_split(np.random.default_rng(9137).permutation(n), 3)
    mass, signed_count, shifted = 0, 0, 0
    for fold, reference in enumerate(folds):
        center = (
            np.zeros(features.shape[1])
            if known_center
            else np.median(features[reference], axis=0)
        )
        shifted += bool(np.any(center))
        fitted = _fit_prototypes(features[reference] - center, 9138 + fold)
        cells = _assign(features - center, fitted) + len(fitted)
        counts = [
            np.bincount(
                cells[folds[(fold + offset) % 3]], minlength=2 * len(fitted) + 1
            )
            for offset in (1, 2)
        ]
        for pilot, measured in (counts, counts[::-1]):
            favored = pilot > pilot[::-1]
            difference = measured - measured[::-1]
            signed_count += difference[favored].sum()
            mass += difference[favored & (measured > 0)].sum()
    return np.array([mass / (2 * n), signed_count / (2 * n), shifted / 3])


def main():
    result = {}
    for name, known in (("estimated_center", False), ("known_zero_center", True)):
        cases = np.stack(
            [diagnose(iid_null(seed), known_center=known) for seed in range(24)]
        )
        result[name] = {
            "mean_estimated_prevalence": float(cases[:, 0].mean()),
            "mean_absolute_estimated_prevalence": float(np.abs(cases[:, 0]).mean()),
            "mean_count_including_empty_selected_cells": float(cases[:, 1].mean()),
            "fraction_fold_centers_displaced": float(cases[:, 2].mean()),
            "prevalence_at_or_below_screening_cutoff": int(
                (cases[:, 0] <= 2 / np.sqrt(2000)).sum()
            ),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
