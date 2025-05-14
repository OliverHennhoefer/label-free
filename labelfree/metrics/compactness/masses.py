import numpy as np
from typing import Any, Callable, Tuple, Union

from labelfree.utils.decorator import as_numpy_array

_rng = np.random.default_rng(seed=1)


# --- Scoring Function ---
def default_scoring_func(model: Any, data: np.ndarray) -> np.ndarray:
    """
    Default scoring function that uses the model's decision_function.

    Args:
        model: A model object with a `decision_function` method.
        data: A NumPy array of data points to score.

    Returns:
        A NumPy array of anomaly scores.
    """
    return model.decision_function(data)


@as_numpy_array("x")
def emmv_scores(
    model: Any,
    x: np.ndarray,  # Decorator ensures this is np.ndarray inside the function
    scoring_func: Callable[[Any, np.ndarray], np.ndarray] = default_scoring_func,
    n_generated: int = 100_000,
    alpha_min: float = 0.9,
    alpha_max: float = 0.999,
) -> Tuple[float, float]:
    """
    Calculates Excess Mass (EM) and Mass Volume (MV) scores for a given dataset
    and model.

    The input `x` can be a pandas DataFrame, Polars DataFrame, or NumPy array.
    It will be converted to a NumPy array internally by the `as_numpy_array`
    decorator for processing.

    Args:
        model: The anomaly detection model (must have a `decision_function` method
               if `default_scoring_func` is used, or be compatible with `scoring_func`).
        x: The input data (features) as a pandas DataFrame, Polars DataFrame, or NumPy array.
        scoring_func: A function that takes the model and data (as NumPy array)
                      and returns anomaly scores (as NumPy array).
                      Defaults to `default_scoring_func`.
        n_generated: Number of synthetic samples to generate for uniform sampling.
        alpha_min: Minimum alpha quantile for MV calculation.
        alpha_max: Maximum alpha quantile for MV calculation.

    Returns:
        A tuple containing:
        - Mean Excess Mass score (float)
        - Mean Mass Volume score (float)
    """
    # Calculate limits, volume, and levels for uniform sampling
    lim_inf, lim_sup, volume, levels = calculate_limits(x)

    # Perform uniform sampling
    # x is guaranteed to be a NumPy array here by the decorator
    if x.ndim == 1:
        # Handle 1D array case for uniform sampling
        # lim_inf and lim_sup will be scalars in this case
        uniform_sample_shape = (n_generated,)
    elif x.ndim > 1:
        uniform_sample_shape = (n_generated, x.shape[1])
    else:  # Should not happen for typical array inputs
        raise ValueError(f"Input array x has unexpected ndim: {x.ndim}")

    uniform_sample = _rng.uniform(lim_inf, lim_sup, size=uniform_sample_shape)

    # If lim_inf/lim_sup were scalars (from 1D input x) and uniform_sample_shape was (n_generated,),
    # uniform_sample will be 1D. If x was NxD, lim_inf/sup are 1D, and uniform_sample is 2D.
    # If uniform_sample is 1D of shape (n_generated,) and scoring_func expects 2D (n_samples, n_features=1),
    # it might need reshaping.
    if uniform_sample.ndim == 1 and x.ndim > 1 and x.shape[1] == 1:  # If x was (N,1)
        uniform_sample = uniform_sample.reshape(-1, 1)
    elif uniform_sample.ndim == 1 and x.ndim == 1:  # If x was (N,)
        # scoring_func needs to handle 1D input for 1D features
        pass

    # Get anomaly scores
    uniform_scores = scoring_func(model, uniform_sample)
    anomaly_scores = scoring_func(model, x)

    # Ensure scores are 1D arrays
    uniform_scores = np.asarray(uniform_scores).ravel()
    anomaly_scores = np.asarray(anomaly_scores).ravel()

    # Calculate and return mean EM and MV scores
    em_values = excess_mass(levels, volume, uniform_scores, anomaly_scores)
    mv_values = mass_volume(
        alpha_min, alpha_max, volume, uniform_scores, anomaly_scores
    )

    return float(np.mean(em_values)), float(np.mean(mv_values))


def calculate_limits(
    x: np.ndarray, offset: float = 1e-60
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], float, np.ndarray]:
    """
    Calculates the bounding box limits, volume, and levels for EM/MV.

    Args:
        x: Input data as a NumPy array.
        offset: A small offset to add to the volume to prevent division by zero
                if features have no variance (all points identical in a dimension).

    Returns:
        A tuple containing:
        - lim_inf: Lower bounds for each feature (or scalar if 1D).
        - lim_sup: Upper bounds for each feature (or scalar if 1D).
        - volume: Volume of the bounding hyperrectangle.
        - levels: Levels at which to evaluate Excess Mass.
    """
    # Min and max values of each feature (per column if x is 2D)
    lim_inf = x.min(axis=0)
    lim_sup = x.max(axis=0)

    # Volume of the bounding rectangle
    # np.prod works fine even if (lim_sup - lim_inf) is scalar (from 1D x)
    volume = float(np.prod(lim_sup - lim_inf))
    if volume == 0:  # Add offset only if volume is truly zero
        volume = offset
    else:  # Add a small relative offset to avoid issues if volume is tiny but non-zero
        volume += offset * volume + offset

    # Levels to evaluate EM_s(t)
    # Ensure levels step is not zero if volume is very large
    step = max(1e-8, 0.01 / volume)  # Avoid step being too small or zero
    upper_bound = max(step * 2, 100.0 / volume)  # Ensure upper_bound > 0
    levels = np.arange(0, upper_bound, step)
    if len(levels) == 0:  # Safety for extreme volume cases
        levels = np.array([0.0, upper_bound / 2.0, upper_bound])

    return lim_inf, lim_sup, volume, levels


def excess_mass(
    levels: np.ndarray,
    volume: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
) -> np.ndarray:
    """
    Calculates Excess Mass (EM) values.
    EM_s(t) = P(S_X > s) - (t/V_R) * P(S_U > s) * V_R where t = level * V_R
    So, EM_s(t) = P(S_X > s) - level * P(S_U > s) * V_R
    No, the definition is E_m(t) = F_X(t) - F_U(t), where F is CDF of scores.
    Or, more commonly with anomaly scores (higher is more anomalous):
    EM_s(score_threshold) = P(S_X > score_threshold) - (Volume_Uniform(score_threshold) / Total_Volume_Uniform_Sample_Space)
    The provided code seems to use:
    EM = max_s { P(S_X > s) - level_factor * P(S_U > s) }
    where `level_factor` is `levels * volume`.

    Let's stick to the paper's definition from "Volume-based Anomaly Detection":
    EM_s(t) = sum_{i=1 to N_X} I(s_X_i > t) / N_X  -  (A_U(t) / V_R)
    where A_U(t) is the volume of the region where uniform samples have score > t.
    A_U(t) / V_R is estimated by P(S_U > t).
    So, EM_s(t) = P(S_X > t) - P(S_U > t)
    The `levels` parameter in the original code is confusing.
    Let's assume `levels` are score thresholds for `t`.

    Re-interpreting the original `excess_mass` function's logic:
    It iterates through unique `anomaly_scores` (`s` in my notation).
    For each `s`:
      `anomaly_fraction = P(S_X > s)`
      `p_uniform_gt_s = P(S_U > s)`
      It then computes `anomaly_fraction - (levels * p_uniform_gt_s * volume)`
      And takes `np.maximum` over these computations for all `s`.
    This means `excess_mass_scores` will have the same shape as `levels`.
    The `levels` array seems to represent different scaling factors for the uniform part.
    This is a non-standard EM formulation if `levels` are not score thresholds.
    If `levels` are indeed `t/V_R` as suggested by `levels = np.arange(0, 100 / volume, 0.01 / volume)`,
    then `uniform * volume` becomes `(t/V_R) * P(S_U > score) * V_R = t * P(S_U > score)`.
    Let's assume the original implementation is what's desired, despite potential naming confusion.

    Args:
        levels: Array of levels (scaling factors for the uniform distribution's contribution).
        volume: Volume of the bounding hyperrectangle of `x`.
        uniform_scores: Anomaly scores of the uniformly sampled data.
        anomaly_scores: Anomaly scores of the input data `x`.

    Returns:
        NumPy array of Excess Mass scores, one for each input level.
    """
    n_samples_x = anomaly_scores.shape[0]
    n_samples_uniform = uniform_scores.shape[0]

    if n_samples_x == 0:
        return np.zeros_like(levels)
    if n_samples_uniform == 0:  # Should not happen with n_generated > 0
        # If no uniform samples, P(S_U > score) is undefined or 0.
        # This would make EM = P(S_X > score), which is not standard.
        # For robustness, return array of zeros or handle as error.
        print("Warning: No uniform scores provided for excess_mass calculation.")
        return np.zeros_like(levels)

    unique_anomaly_thresholds = np.unique(anomaly_scores)
    # Sort thresholds from high to low, as we often care about P(S > s)
    unique_anomaly_thresholds = np.sort(unique_anomaly_thresholds)[::-1]

    current_max_em = np.full(
        levels.shape[0], -np.inf
    )  # Store the max EM found so far for each level
    current_max_em[0] = 1.0  # Matches original if levels[0] = 0

    for score_threshold in unique_anomaly_thresholds:
        # P(S_X > s)
        prob_anomaly_gt_threshold = (
            anomaly_scores > score_threshold
        ).sum() / n_samples_x

        # P(S_U > s)
        prob_uniform_gt_threshold = (
            uniform_scores > score_threshold
        ).sum() / n_samples_uniform

        # Term related to uniform distribution contribution, scaled by levels and volume
        uniform_contribution = levels * prob_uniform_gt_threshold * volume

        # Candidate EM values for the current score_threshold, across all levels
        candidate_em_values = prob_anomaly_gt_threshold - uniform_contribution

        current_max_em = np.maximum(current_max_em, candidate_em_values)

    # Ensure non-negativity if EM is defined as such (often it is, e.g. for ROC like curves)
    # The original code does not enforce this beyond the initial `excess_mass_scores[0] = 1.0`.
    # And `np.maximum` can result in negative if `anomaly_fraction` is small.
    # However, some EM definitions can be negative. Sticking to original behavior.
    return current_max_em


def mass_volume(
    alpha_min: float,
    alpha_max: float,
    volume: float,
    uniform_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    alpha_count: int = 1000,
) -> np.ndarray:
    """
    Calculates Mass Volume (MV) scores.
    For a given mass `alpha`, MV is the volume of the data region containing `alpha`
    fraction of the data points with the highest scores, estimated using uniform samples.

    Args:
        alpha_min: Minimum mass (quantile) to consider.
        alpha_max: Maximum mass (quantile) to consider.
        volume: Volume of the bounding hyperrectangle of original data `x`.
        uniform_scores: Anomaly scores of the uniformly sampled data.
        anomaly_scores: Anomaly scores of the input data `x`.
        alpha_count: Number of alpha values to evaluate between alpha_min and alpha_max.

    Returns:
        NumPy array of Mass Volume scores for each alpha.
    """
    n_samples_x = anomaly_scores.shape[0]
    n_samples_uniform = uniform_scores.shape[0]

    if n_samples_x == 0:
        return np.zeros(alpha_count)  # Or handle as error
    if n_samples_uniform == 0:
        print("Warning: No uniform scores provided for mass_volume calculation.")
        return np.zeros(alpha_count)

    # Sort anomaly scores in ascending order to easily find quantiles
    # argsort gives indices that would sort the array.
    # So anomaly_scores[sorted_indices] is sorted.
    # We need scores corresponding to high mass, so we iterate from highest score downwards.
    sorted_anomaly_scores_desc = np.sort(anomaly_scores)[::-1]  # Highest score first

    alpha_quantiles = np.linspace(alpha_min, alpha_max, alpha_count)
    mv_scores = np.zeros(alpha_quantiles.shape[0])

    # The score_threshold `u` is such that P(S_X > u) approx `mass`
    # Or more precisely, u is the (1-mass)-th quantile of scores if sorted ascending.
    # Since we sorted descending, u is the (mass)-th quantile.
    # score_threshold = sorted_anomaly_scores_desc[0] # Initial highest score

    for i, target_alpha in enumerate(alpha_quantiles):
        # Find the score threshold `u` from `anomaly_scores` such that
        # approximately `target_alpha` fraction of `anomaly_scores` are >= `u`.
        # (or > u, depending on definition, typically >= for sample quantiles)

        # We need at least ceil(target_alpha * n_samples_x) points
        num_points_for_target_alpha = np.ceil(target_alpha * n_samples_x).astype(int)
        # Ensure it's at least 1 if target_alpha > 0, and not beyond n_samples_x
        num_points_for_target_alpha = min(
            max(1, num_points_for_target_alpha), n_samples_x
        )

        # The score threshold is the (num_points_for_target_alpha-1)-th element
        # in the descending sorted scores (0-indexed).
        # E.g., if num_points = 1, we take score_threshold = sorted_scores[0] (highest)
        # If num_points = n_samples_x, we take score_threshold = sorted_scores[n_samples_x-1] (lowest)
        score_threshold_u = sorted_anomaly_scores_desc[num_points_for_target_alpha - 1]

        # mass_check = (anomaly_scores >= score_threshold_u).sum() / n_samples_x
        # This mass_check should be >= target_alpha

        # Calculate fraction of uniform_scores >= score_threshold_u
        # This is P_hat(S_U >= u)
        prob_uniform_ge_threshold = (
            uniform_scores >= score_threshold_u
        ).sum() / n_samples_uniform

        # MV score is this probability scaled by the total volume
        mv_scores[i] = prob_uniform_ge_threshold * volume

    return mv_scores
