"""Internal, Relative Evaluation of Outlier Solutions (IREOS).

Implements Equation 9 and Algorithm 1 of Marques, Campello, Sander, and Zimek
("Internal Evaluation of Unsupervised Outlier Detection", ACM TKDD 2020),
using the authors' Java implementation (homarques/ireos-extension) to resolve
numerical details, and the index's adjustment for chance.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit

from labelfree._validation import as_1d_finite, as_2d_finite, check_integer

_MAX_GAMMA_STEPS = 1_000
_MAX_QUADRATURE_DEPTH = 20
_MAX_NEWTON_STEPS = 200
_NEWTON_TOL = 1e-14
_EIGEN_RTOL = 1e-12
_BIAS_RIDGE = 1e-12
_TIE_TOL = 1e-12


def ireos_score(
    X,
    outlier_probabilities,
    *,
    gamma_max: float | None = None,
    max_clump_size: int = 1,
    penalty_cost: float = 100.0,
    integration_tol: float = 0.005,
) -> float:
    """IREOS index of one outlier solution. Higher is better.

    ``IREOS = sum_j w_j * AUC_j / (gamma_max * sum_j w_j)``, where ``w_j`` is
    the outlier probability of sample ``j`` and ``AUC_j`` integrates, over the
    kernel width ``gamma`` in ``[0, gamma_max]``, the probability that a
    kernel logistic regression classifier assigns to ``j`` when it is
    trained to separate ``j`` from all other samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix. Scale features first.
    outlier_probabilities : array-like of shape (n_samples,)
        Outlier probabilities in ``[0, 1]``, for example from
        :func:`labelfree.unify_scores`. Binary top-n solutions are valid.
    gamma_max : float, optional
        Upper kernel width. By default, the smallest width on the authors'
        geometric grid (start ``1 / (1000 * n_features)``, factor 1.1) that
        separates every sample with probability at least 0.5.
    max_clump_size : int
        Maximum clump size ``mCl``. Sample costs become
        ``penalty_cost * (1 / mCl) ** w_j`` so that small groups of likely
        outliers can be separated together. The authors suggest
        ``round(sqrt(0.05 * n_samples))`` when no domain knowledge exists.
    penalty_cost : float
        Soft-margin penalty ``C``; the authors use 100.
    integration_tol : float
        Absolute tolerance of the adaptive Simpson quadrature.

    Returns
    -------
    float
        IREOS in ``[0, 1]``.
    """
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
    """IREOS of several solutions under one shared ``gamma_max``.

    Comparing solutions requires one kernel-width range, so the automatic
    ``gamma_max`` covers the candidates of every row. Parameters match
    :func:`ireos_score`; ``outlier_probability_matrix`` has shape
    ``(n_solutions, n_samples)``.

    Returns
    -------
    numpy.ndarray of shape (n_solutions,)
        IREOS per solution. Higher is better.
    """
    X, probabilities = _validate_inputs(
        X,
        outlier_probability_matrix,
        gamma_max=gamma_max,
        max_clump_size=max_clump_size,
        penalty_cost=penalty_cost,
        integration_tol=integration_tol,
    )
    kernel_features = _KernelFeatures(cdist(X, X, metric="sqeuclidean"))
    if gamma_max is None:
        gamma_max = _find_gamma_max(
            kernel_features,
            probabilities,
            n_features=X.shape[1],
            penalty_cost=penalty_cost,
        )
    gamma_max = float(gamma_max)

    if max_clump_size == 1:
        candidates = np.flatnonzero(np.any(probabilities > 0, axis=0))
        aucs = _candidate_aucs(
            kernel_features,
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
                kernel_features,
                weights,
                gamma_max=gamma_max,
                max_clump_size=max_clump_size,
                penalty_cost=penalty_cost,
                integration_tol=integration_tol,
            )
            for weights in probabilities
        ]
    )


@dataclass(frozen=True)
class IREOSSignificance:
    """IREOS of one solution with its chance-adjusted value and significance.

    Attributes
    ----------
    index : float
        Raw IREOS.
    expected : float
        Expected IREOS of a random solution with the same probabilities
        assigned to random samples.
    std : float
        Standard deviation of IREOS under that null model.
    adjusted : float
        ``(index - expected) / (1 - expected)``; 0 is chance level, 1 is the
        maximum.
    z_score : float
        ``(index - expected) / std``.
    p_value : float
        One-sided probability that a random solution scores at least
        ``index``.
    method : str
        ``"exact_moments"`` for ``max_clump_size=1`` (exact ``expected`` and
        ``std``); ``"monte_carlo"`` otherwise. The p-value is a permutation
        p-value in both cases.
    """

    index: float
    expected: float
    std: float
    adjusted: float
    z_score: float
    p_value: float
    method: str


def ireos_significance(
    X,
    outlier_probabilities,
    *,
    gamma_max: float | None = None,
    max_clump_size: int = 1,
    penalty_cost: float = 100.0,
    integration_tol: float = 0.005,
    n_permutations: int | None = None,
    random_state=None,
) -> IREOSSignificance:
    """Adjust IREOS for chance and test it against random solutions.

    Marques et al. adjust IREOS for chance in the form
    ``(I - E[I]) / (max I - E[I])`` with ``max I = 1``, and validate a
    solution statistically against randomly labeled solutions. Here the null
    model assigns the solution's outlier probabilities to random samples, that
    is, it permutes ``outlier_probabilities``.

    With ``max_clump_size=1`` the separability curves do not depend on the
    solution, so IREOS is a linear permutation statistic and its null mean and
    variance are exact:
    ``E[I] = mean(AUC) / gamma_max`` and
    ``Var[I] = sum((w - mean w)^2) * sum((AUC - mean AUC)^2)
    / ((n - 1) * (gamma_max * sum w)^2)``. Random solutions reuse the same
    curves, so permutations are cheap. This needs one separability curve per
    sample.

    With ``max_clump_size > 1`` the costs depend on the solution, so the null
    mean and deviation are estimated from the permuted solutions, each of
    which costs as much as one IREOS evaluation.

    The p-value is always the permutation p-value
    ``(1 + #{I_perm >= I}) / (1 + n_permutations)``, which stays valid for the
    small, discrete null distributions of sparse or binary solutions where a
    normal approximation would be anti-conservative. Its resolution is
    ``1 / (1 + n_permutations)``.

    Parameters
    ----------
    X, outlier_probabilities, gamma_max, max_clump_size, penalty_cost, integration_tol
        As in :func:`ireos_score`. The automatic ``gamma_max`` is chosen from
        the observed solution and reused for every random solution.
    n_permutations : int, optional
        Number of random solutions. Defaults to 10,000 for
        ``max_clump_size=1`` and 100 otherwise.
    random_state : int, numpy.random.Generator, or None
        Seed for the permutations.

    Returns
    -------
    IREOSSignificance
    """
    probabilities = as_1d_finite(outlier_probabilities, name="outlier_probabilities")
    X, matrix = _validate_inputs(
        X,
        probabilities[None, :],
        gamma_max=gamma_max,
        max_clump_size=max_clump_size,
        penalty_cost=penalty_cost,
        integration_tol=integration_tol,
    )
    weights = matrix[0]
    kernel_features = _KernelFeatures(cdist(X, X, metric="sqeuclidean"))
    if gamma_max is None:
        gamma_max = _find_gamma_max(
            kernel_features,
            matrix,
            n_features=X.shape[1],
            penalty_cost=penalty_cost,
        )
    gamma_max = float(gamma_max)
    if n_permutations is None:
        n_permutations = 10_000 if max_clump_size == 1 else 100
    check_integer(n_permutations, name="n_permutations")
    if n_permutations < 2:
        raise ValueError("n_permutations must be at least 2")
    rng = np.random.default_rng(random_state)
    settings = {
        "gamma_max": gamma_max,
        "max_clump_size": max_clump_size,
        "penalty_cost": penalty_cost,
        "integration_tol": integration_tol,
    }

    if max_clump_size == 1:
        aucs = _candidate_aucs(
            kernel_features,
            np.arange(X.shape[0]),
            np.full(X.shape[0], penalty_cost),
            gamma_max=gamma_max,
            penalty_cost=penalty_cost,
            integration_tol=integration_tol,
        )
        scale = gamma_max * weights.sum()
        index = float(weights @ aucs / scale)
        expected = float(aucs.mean() / gamma_max)
        variance = (
            np.sum((weights - weights.mean()) ** 2)
            * np.sum((aucs - aucs.mean()) ** 2)
            / ((X.shape[0] - 1) * scale**2)
        )
        std = float(np.sqrt(variance))
        null = np.array(
            [rng.permutation(weights) @ aucs / scale for _ in range(n_permutations)]
        )
        method = "exact_moments"
    else:
        index = _score_solution(kernel_features, weights, **settings)
        null = np.array(
            [
                _score_solution(kernel_features, rng.permutation(weights), **settings)
                for _ in range(n_permutations)
            ]
        )
        expected = float(null.mean())
        std = float(null.std(ddof=1))
        method = "monte_carlo"

    z_score = _z_score(index, expected, std)
    at_least = np.count_nonzero(null >= index - _TIE_TOL * max(1.0, abs(index)))
    p_value = float((1 + at_least) / (1 + n_permutations))

    adjusted = (index - expected) / (1 - expected) if expected < 1 else 0.0
    return IREOSSignificance(
        index=index,
        expected=expected,
        std=std,
        adjusted=float(adjusted),
        z_score=z_score,
        p_value=p_value,
        method=method,
    )


def _z_score(index: float, expected: float, std: float) -> float:
    if std > 0:
        return float((index - expected) / std)
    return (
        0.0
        if np.isclose(index, expected)
        else float(np.sign(index - expected) * np.inf)
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
    kernel_features: _KernelFeatures,
    probability_matrix: np.ndarray,
    *,
    n_features: int,
    penalty_cost: float,
) -> float:
    """Follow ``Utils.findGammaMax`` of the authors' implementation.

    Candidates are samples with outlier probability of at least 0.5 in any
    solution. Starting at ``1 / (1000 * n_features)``, gamma grows by 10% until
    the classifier, trained with uniform costs, gives the current candidate a
    probability above 0.5; separated candidates are then dropped.
    """
    pending = list(np.flatnonzero(np.any(probability_matrix >= 0.5, axis=0)))
    if not pending:
        raise ValueError(
            "automatic gamma_max requires an outlier probability of at least 0.5; "
            "pass gamma_max explicitly"
        )

    costs = np.full(kernel_features.squared_distances.shape[0], penalty_cost)
    gamma = 1.0 / (n_features * 1_000)
    gamma_steps = 0
    while pending:
        probability = _kernel_logistic_probability(
            kernel_features, gamma, pending[0], costs
        )
        if probability > 0.5:
            pending.pop(0)
            continue
        gamma *= 1.1
        gamma_steps += 1
        if gamma_steps >= _MAX_GAMMA_STEPS or not np.isfinite(gamma):
            raise RuntimeError(
                "automatic gamma_max search did not separate every candidate; "
                "scale X or pass gamma_max explicitly"
            )
    return gamma


def _score_solution(
    kernel_features: _KernelFeatures,
    weights: np.ndarray,
    *,
    gamma_max: float,
    max_clump_size: int,
    penalty_cost: float,
    integration_tol: float,
) -> float:
    costs = penalty_cost * np.power(1.0 / max_clump_size, weights)
    aucs = _candidate_aucs(
        kernel_features,
        np.flatnonzero(weights > 0),
        costs,
        gamma_max=gamma_max,
        penalty_cost=penalty_cost,
        integration_tol=integration_tol,
    )
    return float(np.dot(weights, aucs) / (weights.sum() * gamma_max))


def _candidate_aucs(
    kernel_features: _KernelFeatures,
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
                    kernel_features,
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


class _KernelFeatures:
    """Low-rank feature maps ``Phi`` with ``Phi @ Phi.T == exp(-gamma * D2)``.

    Writing ``f = Phi @ beta + b`` turns the regularizer ``0.5 * a' K a`` into
    ``0.5 * ||beta||^2``, which keeps Newton's method well conditioned even
    when the kernel matrix is numerically singular (small ``gamma``). Feature
    maps are cached per ``gamma`` because candidates share quadrature nodes.
    """

    def __init__(self, squared_distances: np.ndarray, max_bytes: int = 2**28):
        self.squared_distances = squared_distances
        n_samples = squared_distances.shape[0]
        self.capacity = max(1, max_bytes // (8 * n_samples * n_samples))
        self.cache: OrderedDict[float, np.ndarray] = OrderedDict()

    def __call__(self, gamma: float) -> np.ndarray:
        if gamma in self.cache:
            self.cache.move_to_end(gamma)
            return self.cache[gamma]
        kernel = np.exp(-gamma * self.squared_distances)
        eigenvalues, eigenvectors = np.linalg.eigh(kernel)
        keep = eigenvalues > eigenvalues[-1] * _EIGEN_RTOL
        features = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        self.cache[gamma] = features
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return features


def _kernel_logistic_probability(
    kernel_features,
    gamma: float,
    candidate: int,
    costs: np.ndarray,
) -> float:
    """Probability of ``candidate`` under kernel logistic regression.

    Minimizes ``0.5 * a' K a + sum_i c_i * log(1 + exp(-y_i * f_i))`` with
    ``f = K a + b``, ``y = +1`` for the candidate and ``-1`` otherwise, by
    Newton's method in the feature space of ``K``. ``kernel_features`` is a
    :class:`_KernelFeatures` or a squared-distance matrix.
    """
    if not isinstance(kernel_features, _KernelFeatures):
        kernel_features = _KernelFeatures(np.asarray(kernel_features, dtype=float))
    design = np.hstack([kernel_features(gamma), np.ones((costs.size, 1))])
    labels = -np.ones(costs.size)
    labels[candidate] = 1.0
    penalized = np.ones(design.shape[1])
    penalized[-1] = 0.0

    def objective(theta: np.ndarray) -> float:
        margin = labels * (design @ theta)
        return 0.5 * float(theta[:-1] @ theta[:-1]) + float(
            costs @ np.logaddexp(0.0, -margin)
        )

    theta = np.zeros(design.shape[1])
    value = objective(theta)
    for _ in range(_MAX_NEWTON_STEPS):
        margin = labels * (design @ theta)
        probability = expit(margin)
        gradient = penalized * theta - design.T @ (costs * labels * (1 - probability))
        hessian = (design.T * (costs * probability * (1 - probability))) @ design
        hessian[np.diag_indices_from(hessian)] += penalized + _BIAS_RIDGE
        step = np.linalg.solve(hessian, -gradient)
        decrement = -float(gradient @ step)
        if decrement / 2 <= _NEWTON_TOL * max(1.0, abs(value)):
            break
        size = 1.0
        while True:
            candidate_theta = theta + size * step
            candidate_value = objective(candidate_theta)
            if candidate_value <= value - 0.25 * size * decrement or size < 1e-10:
                break
            size /= 2
        theta, value = candidate_theta, candidate_value
    else:
        raise RuntimeError("kernel logistic regression did not converge")
    return float(expit(design[candidate] @ theta))


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
