"""Falsification of a direct empirical use of the ordered-cohort population bound.

Registered before this simulation. Use the referee's two detectors A and B in
its b=24 world (A wins). Split an irrelevant cell into a common cell and a rare
cell with mass 1/n in both cohorts, leaving detector scores unchanged. Predict
that the empirical fine-partition bound excludes the true b when the rare cell
occurs only in the target. Predict a false certificate for B on over 15% of all
draws and fewer such certificates on the original three score cells. This tests
plug-in inference. The population theorem uses the true cell probabilities.
"""

import numpy as np


def declares_b_winner(target, enriched):
    difference = enriched - target
    feasible = ~np.any((target == 0) & (enriched > 0), axis=1)
    ratio = np.full_like(target, np.inf)
    np.divide(target, -difference, out=ratio, where=difference < 0)
    upper = ratio.min(axis=1)
    # For a binary detector that flags cell i the coefficients are
    # L_i = D_i and Q_i = D_i**2 / T_i, with all other cells at the lower score.
    linear = difference[:, 0] - difference[:, 1]
    quadratic = (
        difference[:, 0] ** 2 / target[:, 0] - difference[:, 1] ** 2 / target[:, 1]
    )
    return feasible & (linear + quadratic < 0) & (linear + upper * quadratic < 0)


def main():
    n, repeats = 1_000_000, 4096
    rare = 1 / n
    target = [0.05, 0.50, 0.45 - rare, rare]
    enriched = [0.055, 0.51, 0.435 - rare, rare]
    rng = np.random.default_rng(1172)
    target_sample = rng.multinomial(n, target, size=repeats) / n
    enriched_sample = rng.multinomial(n, enriched, size=repeats) / n
    only_in_target = (target_sample[:, 3] > 0) & (enriched_sample[:, 3] == 0)
    absent = (1 - rare) ** n
    fine = declares_b_winner(target_sample, enriched_sample)
    # Both detectors assign identical scores to the last two cells.
    coarse = declares_b_winner(
        np.column_stack((target_sample[:, :2], target_sample[:, 2:].sum(axis=1))),
        np.column_stack((enriched_sample[:, :2], enriched_sample[:, 2:].sum(axis=1))),
    )
    print(
        f"P(target-only rare cell) exact {absent * (1 - absent):.6f}, simulated {only_in_target.mean():.6f}"
    )
    print(
        f"False certificates for B in {repeats} draws: fine {fine.sum()}, coarse {coarse.sum()}"
    )


if __name__ == "__main__":
    main()
