"""Internal, Relative Evaluation of Outlier Solutions (IREOS).

Implements Equation 9 and Algorithm 1 of Marques et al. (2020), using the
authors' Java implementation to resolve numerical details.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

from labelfree.utils.validation import as_1d_finite, as_2d_finite

_MAX_GAMMA_STEPS = 1_000
_MAX_QUADRATURE_DEPTH = 20


def ireos_score(
    X,
    outlier_probabilities,
    *,
    gamma_max: float | None = None,
    max_clump_size: int = 1,
    penalty_cost: float = 100.0,
    integration_tol: float = 0.005,
) -> float:
    """Evaluate one normalized outlier-probability vector. Higher is better."""
    probabilities = as_1d_finite(outlier_probabilities, name="outlier_probabilities")
    return float(
        ireos_scores(
            X,
            probabilities[None, :],
            gamma_max=gamma_max,
            max_clump_size=max_clump_size,
            penalty_cost=penalty_cost,
            integration_tol=integration_tol,
        )[0]
    )


def ireos_scores(
    X,
    outlier_probability_matrix,
    *,
    gamma_max: float | None = None,
    max_clump_size: int = 1,
    penalty_cost: float = 100.0,
    integration_tol: float = 0.005,
) -> np.ndarray:
    """Evaluate probability rows using one shared IREOS gamma range."""
    X, probabilities = _validate_inputs(
        X,
        outlier_probability_matrix,
        gamma_max=gamma_max,
        max_clump_size=max_clump_size,
        penalty_cost=penalty_cost,
        integration_tol=integration_tol,
    )
    squared_distances = cdist(X, X, metric="sqeuclidean")
    if gamma_max is None:
        gamma_max = _find_gamma_max(
            squared_distances,
            probabilities,
            n_features=X.shape[1],
            max_clump_size=max_clump_size,
            penalty_cost=penalty_cost,
        )
    gamma_max = float(gamma_max)

    if max_clump_size == 1:
        candidates = np.flatnonzero(np.any(probabilities > 0, axis=0))
        aucs = _candidate_aucs(
            squared_distances,
            candidates,
            np.full(X.shape[0], penalty_cost),
            gamma_max=gamma_max,
            penalty_cost=penalty_cost,
            integration_tol=integration_tol,
        )
        return (probabilities @ aucs) / (probabilities.sum(axis=1) * gamma_max)

    return np.array(
        [
            _score_solution(
                squared_distances,
                weights,
                gamma_max=gamma_max,
                max_clump_size=max_clump_size,
                penalty_cost=penalty_cost,
                integration_tol=integration_tol,
            )
            for weights in probabilities
        ]
    )


def _validate_inputs(
    X,
    probability_matrix,
    *,
    gamma_max: float | None,
    max_clump_size: int,
    penalty_cost: float,
    integration_tol: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = as_2d_finite(X, name="X")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two samples")

    probabilities = np.asarray(probability_matrix, dtype=float)
    if probabilities.ndim != 2 or 0 in probabilities.shape:
        raise ValueError("outlier_probability_matrix must be a non-empty 2D array")
    if not np.isfinite(probabilities).all():
        raise ValueError("outlier_probability_matrix contains non-finite values")
    if probabilities.shape[1] != X.shape[0]:
        raise ValueError("X and outlier probabilities must have the same sample count")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("outlier probabilities must be between 0 and 1")
    if np.any(probabilities.sum(axis=1) == 0):
        raise ValueError("each outlier solution must have positive total weight")

    if isinstance(max_clump_size, bool) or not isinstance(
        max_clump_size, (int, np.integer)
    ):
        raise ValueError("max_clump_size must be an integer")
    if max_clump_size < 1:
        raise ValueError("max_clump_size must be positive")
    for weights in probabilities:
        if np.all((weights == 0) | (weights == 1)):
            if max_clump_size > np.count_nonzero(weights):
                raise ValueError(
                    "max_clump_size cannot exceed the number of binary outliers"
                )

    for value, name in (
        (penalty_cost, "penalty_cost"),
        (integration_tol, "integration_tol"),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive value")
    if gamma_max is not None and (not np.isfinite(gamma_max) or gamma_max <= 0):
        raise ValueError("gamma_max must be a finite positive value")
    return X, probabilities


def _find_gamma_max(
    squared_distances: np.ndarray,
    probability_matrix: np.ndarray,
    *,
    n_features: int,
    max_clump_size: int,
    penalty_cost: float,
) -> float:
    if max_clump_size == 1:
        candidates = [
            (probability_matrix[0], index)
            for index in np.flatnonzero(np.any(probability_matrix > 0.5, axis=0))
        ]
    else:
        candidates = [
            (weights, index)
            for weights in probability_matrix
            for index in np.flatnonzero(weights > 0.5)
        ]
    if not candidates:
        raise ValueError(
            "automatic gamma_max requires an outlier probability above 0.5; "
            "pass gamma_max explicitly"
        )

    pending = candidates.copy()
    gamma = 1.0 / (n_features * 1_000)
    gamma_steps = 0

    def candidate_probability(weights, candidate):
        costs = penalty_cost * np.power(1.0 / max_clump_size, weights)
        costs[candidate] = penalty_cost
        return _kernel_logistic_probability(
            squared_distances,
            gamma,
            candidate,
            costs,
        )

    while pending:
        weights, candidate = pending[-1]
        probability = candidate_probability(weights, candidate)
        if probability > 0.5:
            pending.pop()
            if not pending:
                pending = [
                    (candidate_weights, candidate_index)
                    for candidate_weights, candidate_index in candidates
                    if candidate_probability(candidate_weights, candidate_index) <= 0.5
                ]
                if not pending:
                    return gamma
        else:
            gamma *= 1.1
            gamma_steps += 1
            if gamma_steps >= _MAX_GAMMA_STEPS or not np.isfinite(gamma):
                break
    raise RuntimeError(
        "automatic gamma_max search did not separate every candidate; "
        "scale X or pass gamma_max explicitly"
    )


def _score_solution(
    squared_distances: np.ndarray,
    weights: np.ndarray,
    *,
    gamma_max: float,
    max_clump_size: int,
    penalty_cost: float,
    integration_tol: float,
) -> float:
    costs = penalty_cost * np.power(1.0 / max_clump_size, weights)
    aucs = _candidate_aucs(
        squared_distances,
        np.flatnonzero(weights > 0),
        costs,
        gamma_max=gamma_max,
        penalty_cost=penalty_cost,
        integration_tol=integration_tol,
    )
    return float(np.dot(weights, aucs) / (weights.sum() * gamma_max))


def _candidate_aucs(
    squared_distances: np.ndarray,
    candidates: np.ndarray,
    costs: np.ndarray,
    *,
    gamma_max: float,
    penalty_cost: float,
    integration_tol: float,
) -> np.ndarray:
    aucs = np.zeros(costs.size)
    for candidate in candidates:
        candidate_costs = costs.copy()
        candidate_costs[candidate] = penalty_cost
        cache: dict[float, float] = {}

        def separability(gamma: float) -> float:
            if gamma not in cache:
                cache[gamma] = _kernel_logistic_probability(
                    squared_distances,
                    gamma,
                    candidate,
                    candidate_costs,
                )
            return cache[gamma]

        aucs[candidate] = _adaptive_simpson(
            separability,
            0.0,
            gamma_max,
            integration_tol,
        )
    return aucs


def _kernel_logistic_probability(
    squared_distances: np.ndarray,
    gamma: float,
    candidate: int,
    costs: np.ndarray,
) -> float:
    kernel = np.exp(-gamma * squared_distances)
    labels = -np.ones(costs.size)
    labels[candidate] = 1.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficients = parameters[:-1]
        scores = kernel @ coefficients + parameters[-1]
        negative_margin = -labels * scores
        value = 0.5 * coefficients @ kernel @ coefficients
        value += np.dot(costs, np.logaddexp(0.0, negative_margin))
        loss_gradient = -costs * labels * expit(negative_margin)
        gradient = np.empty_like(parameters)
        gradient[:-1] = kernel @ (coefficients + loss_gradient)
        gradient[-1] = loss_gradient.sum()
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(costs.size + 1),
        method="L-BFGS-B",
        jac=True,
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 2_000},
    )
    if not result.success:
        raise RuntimeError(f"kernel logistic regression failed: {result.message}")
    return float(expit(kernel[candidate] @ result.x[:-1] + result.x[-1]))


def _adaptive_simpson(
    function,
    lower: float,
    upper: float,
    tolerance: float,
) -> float:
    midpoint = (lower + upper) / 2
    f_lower = function(lower)
    f_midpoint = function(midpoint)
    f_upper = function(upper)
    whole = (upper - lower) * (f_lower + 4 * f_midpoint + f_upper) / 6

    def recurse(a, b, fa, fm, fb, estimate, tol, depth):
        middle = (a + b) / 2
        left_middle = (a + middle) / 2
        right_middle = (middle + b) / 2
        f_left_middle = function(left_middle)
        f_right_middle = function(right_middle)
        left = (middle - a) * (fa + 4 * f_left_middle + fm) / 6
        right = (b - middle) * (fm + 4 * f_right_middle + fb) / 6
        refined = left + right
        if abs(estimate - refined) / 15 <= tol:
            return refined
        if depth == 0:
            raise RuntimeError("adaptive Simpson integration did not converge")
        return recurse(
            a,
            middle,
            fa,
            f_left_middle,
            fm,
            left,
            tol / 2,
            depth - 1,
        ) + recurse(
            middle,
            b,
            fm,
            f_right_middle,
            fb,
            right,
            tol / 2,
            depth - 1,
        )

    return float(
        recurse(
            lower,
            upper,
            f_lower,
            f_midpoint,
            f_upper,
            whole,
            tolerance,
            _MAX_QUADRATURE_DEPTH,
        )
    )
