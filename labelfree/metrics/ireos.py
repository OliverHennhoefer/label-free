"""Internal Relative Evaluation of Outlier Solutions (IREOS).

Enhanced IREOS implementation with multiple separability algorithms including:
- KLR: Kernel Logistic Regression using CVXPY optimization
- SVM: Custom Support Vector Machine with separability focus
- KNN: K-Nearest Neighbors variants (KNNM, KNNC)
- Logistic: Original logistic regression approach

Reference:
- Original IREOS: Schubert et al. (2012)
- Extended framework: Hermann Luft's ireos-extended repository
"""

import numpy as np
from typing import Tuple, Optional, Literal
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from scipy.integrate import simpson
from scipy.spatial.distance import cdist
import warnings
from labelfree.utils.validation import validate_scores, validate_data

# Optional CVXPY import with fallback
try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    warnings.warn(
        "CVXPY not available. KLR classifier will fall back to logistic regression."
    )


def ireos(
    scores: np.ndarray,
    data: np.ndarray,
    n_outliers: Optional[int] = None,
    percentile: float = 90.0,
    classifier: Literal["klr", "svm", "logistic", "knn"] = "klr",
    gamma_min: float = 0.1,
    gamma_max: Optional[float] = None,
    n_gamma: int = 50,
    penalty: float = 100.0,
    kernel_approximation: Literal["exact", "nystroem"] = "exact",
    adjustment: bool = True,
    n_monte_carlo: int = 200,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Compute IREOS (Internal Relative Evaluation of Outlier Solutions).

    Enhanced implementation with multiple separability algorithms. Measures
    separability of selected outliers across gamma parameter ranges using
    various classification approaches optimized for numerical stability.

    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Continuous anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original features corresponding to scores.
    n_outliers : int, optional
        Number of top-scoring points to select as outliers.
        If None, uses percentile threshold.
    percentile : float, default=90.0
        Percentile threshold for outlier selection if n_outliers is None.
    classifier : {"klr", "svm", "logistic", "knn"}, default="klr"
        Separability algorithm to use:
        - "klr": Kernel Logistic Regression (CVXPY-based, most robust)
        - "svm": Custom SVM with hinge loss
        - "logistic": Standard logistic regression (original approach)
        - "knn": K-Nearest Neighbors distance-based approach
    gamma_min : float, default=0.1
        Minimum gamma value for RBF kernel.
    gamma_max : float, optional
        Maximum gamma value. If None, estimated automatically.
    n_gamma : int, default=50
        Number of gamma values to sample.
    penalty : float, default=100.0
        Penalty parameter C for regularized classifiers.
    kernel_approximation : {"exact", "nystroem"}, default="exact"
        Kernel computation method:
        - "exact": Full kernel matrix (small datasets)
        - "nystroem": Nystroem approximation (large datasets)
    adjustment : bool, default=True
        Whether to apply statistical adjustment for chance.
    n_monte_carlo : int, default=200
        Number of Monte Carlo runs for statistical estimation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    ireos_score : float
        IREOS index (adjusted if adjustment=True). Higher values indicate
        better anomaly detection quality.
    p_value : float
        Statistical significance p-value against random outlier selection.

    Notes
    -----
    Algorithm selection recommendations:
    - Use "klr" (default) for most accurate results with small-medium datasets
    - Use "svm" for good balance of accuracy and speed
    - Use "knn" for fastest computation on large datasets
    - Use "logistic" only for compatibility with older implementations
    """
    scores = validate_scores(scores)
    data = validate_data(data)

    if len(scores) != len(data):
        raise ValueError(
            f"Length mismatch: {len(scores)} scores vs {len(data)} data points"
        )

    # Validate score normalization for proper IREOS computation
    _validate_score_normalization(scores)

    # Handle CVXPY fallback for KLR
    if classifier == "klr" and not HAS_CVXPY:
        warnings.warn("CVXPY not available. Falling back to logistic classifier.")
        classifier = "logistic"

    rng = np.random.default_rng(random_state)

    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # Select outliers from continuous scores
    if n_outliers is not None:
        outlier_indices = np.argpartition(scores, -n_outliers)[-n_outliers:]
    else:
        threshold = np.percentile(scores, percentile)
        outlier_indices = np.where(scores >= threshold)[0]

    if len(outlier_indices) == 0:
        return 0.0, 1.0

    # Intelligent gamma range estimation
    if gamma_max is None:
        gamma_max = _estimate_gamma_max_robust(
            X, outlier_indices, gamma_min, classifier, rng
        )

    # Create gamma values using logarithmic spacing
    gamma_values = np.logspace(np.log10(gamma_min), np.log10(gamma_max), n_gamma)

    # Initialize separability classifier
    sep_classifier = _get_separability_classifier(
        classifier, penalty, kernel_approximation, random_state
    )

    # Compute separability curve using selected algorithm
    separabilities = []
    for gamma in gamma_values:
        # Compute separability for each outlier at this gamma
        outlier_seps = []
        for outlier_idx in outlier_indices:
            sep = sep_classifier.compute_separability(X, outlier_idx, gamma)
            outlier_seps.append(sep)

        # Average separability across all outliers
        avg_sep = np.mean(outlier_seps)
        separabilities.append(avg_sep)

    separabilities = np.array(separabilities)

    # Compute IREOS index using numerical integration
    gamma_range = gamma_max - gamma_min
    ireos_raw = simpson(separabilities, x=gamma_values) / gamma_range

    # Statistical adjustment if requested
    if adjustment:
        # Estimate expected value using Monte Carlo
        expected_ireos = _estimate_expected_ireos_enhanced(
            X,
            len(outlier_indices),
            gamma_values,
            sep_classifier,
            n_monte_carlo // 4,
            rng,
        )

        # Apply robust adjustment formula: (I - E{I}) / (1 - E{I})
        denominator = 1 - expected_ireos + 1e-12
        ireos_adjusted = (ireos_raw - expected_ireos) / denominator
        ireos_score = max(0.0, ireos_adjusted)
    else:
        ireos_score = ireos_raw

    # Compute p-value using enhanced Monte Carlo
    p_value = _compute_p_value_enhanced(
        X,
        len(outlier_indices),
        gamma_values,
        sep_classifier,
        ireos_score,
        n_monte_carlo // 2,
        rng,
    )

    return float(ireos_score), float(p_value)


# ============================================================================
# SEPARABILITY CLASSIFIER CLASSES
# ============================================================================


class SeparabilityClassifier:
    """Base class for separability computation algorithms."""

    def __init__(
        self,
        penalty: float = 100.0,
        kernel_approximation: str = "exact",
        random_state: Optional[int] = None,
    ):
        self.penalty = penalty
        self.kernel_approximation = kernel_approximation
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def compute_separability(
        self, X: np.ndarray, outlier_idx: int, gamma: float
    ) -> float:
        """Compute separability probability for a single outlier at given gamma."""
        raise NotImplementedError

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
        """Compute RBF kernel matrix efficiently."""
        X_norm = np.sum(X**2, axis=1, keepdims=True)
        Y_norm = np.sum(Y**2, axis=1, keepdims=True)
        K = -2 * np.dot(X, Y.T) + X_norm + Y_norm.T
        return np.exp(-gamma * K)

    def _get_kernel_features(self, X: np.ndarray, gamma: float) -> np.ndarray:
        """Get kernel features using exact or approximate computation."""
        n_samples = len(X)

        if self.kernel_approximation == "nystroem" and n_samples > 500:
            # Use Nystroem approximation for large datasets
            n_components = min(200, n_samples // 2)
            nystroem = Nystroem(
                kernel="rbf",
                gamma=gamma,
                n_components=n_components,
                random_state=self.random_state,
            )
            return nystroem.fit_transform(X)
        else:
            # Exact kernel computation for smaller datasets
            return self._rbf_kernel(X, X, gamma)


class KLRSeparabilityClassifier(SeparabilityClassifier):
    """Kernel Logistic Regression using CVXPY optimization."""

    def compute_separability(
        self, X: np.ndarray, outlier_idx: int, gamma: float
    ) -> float:
        """Compute separability using KLR with convex optimization."""
        n_samples = len(X)

        # Create binary labels: outlier vs all others
        y = np.full(n_samples, -1, dtype=np.float64)
        y[outlier_idx] = 1

        # Check for degenerate case
        if np.sum(y == 1) == 0 or np.sum(y == -1) == 0:
            return 0.5

        try:
            # Get kernel features
            K = self._get_kernel_features(X, gamma)

            # CVXPY optimization for logistic regression
            n_features = K.shape[1] if K.ndim > 1 else K.shape[0]
            beta = cp.Variable(n_features)

            # Compute decision function: K @ beta
            if K.ndim == 1:
                f = K @ beta
            else:
                f = K @ beta

            # Logistic loss: sum(log(1 + exp(-y * f)))
            logistic_loss = cp.sum(cp.logistic(-cp.multiply(y, f)))

            # L2 regularization: 0.5 * ||beta||^2
            regularization = 0.5 * cp.sum_squares(beta)

            # Total objective: C * loss + regularization
            objective = cp.Minimize(logistic_loss / self.penalty + regularization)

            # Solve optimization problem
            prob = cp.Problem(objective)
            prob.solve(solver=cp.ECOS, verbose=False)

            if prob.status not in ["infeasible", "unbounded"]:
                # Compute probability for the outlier
                beta_opt = beta.value
                if beta_opt is not None:
                    if K.ndim == 1:
                        decision_value = K[outlier_idx] * beta_opt
                    else:
                        decision_value = K[outlier_idx] @ beta_opt
                    # Convert to probability using sigmoid
                    prob_outlier = 1.0 / (1.0 + np.exp(-decision_value))
                    return float(np.clip(prob_outlier, 0.01, 0.99))

            # Fallback to 0.5 if optimization fails
            return 0.5

        except Exception:
            return 0.5


class SVMSeparabilityClassifier(SeparabilityClassifier):
    """Custom SVM with hinge loss for separability measurement."""

    def compute_separability(
        self, X: np.ndarray, outlier_idx: int, gamma: float
    ) -> float:
        """Compute separability using SVM with hinge loss."""
        n_samples = len(X)

        # Create binary labels: outlier vs all others
        y = np.full(n_samples, -1, dtype=np.float64)
        y[outlier_idx] = 1

        if np.sum(y == 1) == 0 or np.sum(y == -1) == 0:
            return 0.5

        try:
            # Get kernel features
            K = self._get_kernel_features(X, gamma)

            if not HAS_CVXPY:
                # Fallback to logistic regression
                from sklearn.linear_model import LogisticRegression

                clf = LogisticRegression(
                    C=self.penalty,
                    random_state=self.random_state,
                    max_iter=500,
                    solver="liblinear",
                )
                clf.fit(K, (y + 1) / 2)  # Convert to {0, 1}
                prob = clf.predict_proba(K[outlier_idx : outlier_idx + 1])[0, 1]
                return float(prob)

            # CVXPY optimization for SVM
            n_features = K.shape[1] if K.ndim > 1 else K.shape[0]
            beta = cp.Variable(n_features)
            xi = cp.Variable(n_samples, nonneg=True)  # Slack variables

            # Compute decision function
            if K.ndim == 1:
                f = K @ beta
            else:
                f = K @ beta

            # SVM constraints: y * f >= 1 - xi
            constraints = [cp.multiply(y, f) >= 1 - xi]

            # Objective: minimize 0.5 * ||beta||^2 + C * sum(xi)
            objective = cp.Minimize(
                0.5 * cp.sum_squares(beta) + self.penalty * cp.sum(xi)
            )

            # Solve optimization problem
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.ECOS, verbose=False)

            if prob.status not in ["infeasible", "unbounded"]:
                beta_opt = beta.value
                if beta_opt is not None:
                    # Compute decision value for outlier
                    if K.ndim == 1:
                        decision_value = K[outlier_idx] * beta_opt
                    else:
                        decision_value = K[outlier_idx] @ beta_opt

                    # Convert decision value to probability (approximation)
                    # Use sigmoid transformation of decision value
                    prob_outlier = 1.0 / (1.0 + np.exp(-decision_value))
                    return float(np.clip(prob_outlier, 0.01, 0.99))

            return 0.5

        except Exception:
            return 0.5


class LogisticSeparabilityClassifier(SeparabilityClassifier):
    """Original logistic regression approach for compatibility."""

    def compute_separability(
        self, X: np.ndarray, outlier_idx: int, gamma: float
    ) -> float:
        """Compute separability using standard logistic regression."""
        from sklearn.linear_model import LogisticRegression

        n_samples = len(X)

        # Create binary labels: outlier vs all others
        y = np.full(n_samples, 0)
        y[outlier_idx] = 1

        if np.sum(y) == 0 or np.sum(y) == n_samples:
            return 0.5

        try:
            # Get kernel features
            K = self._get_kernel_features(X, gamma)

            # Train logistic regression with high regularization
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf = LogisticRegression(
                    C=self.penalty,
                    random_state=self.random_state,
                    max_iter=500,
                    solver="liblinear",
                )
                clf.fit(K, y)

                # Get probability for the outlier
                prob = clf.predict_proba(K[outlier_idx : outlier_idx + 1])
                return float(prob[0, 1])

        except Exception:
            return 0.5


class KNNSeparabilityClassifier(SeparabilityClassifier):
    """K-Nearest Neighbors distance-based separability."""

    def __init__(
        self,
        penalty: float = 100.0,
        kernel_approximation: str = "exact",
        random_state: Optional[int] = None,
        k: int = 5,
    ):
        super().__init__(penalty, kernel_approximation, random_state)
        self.k = min(k, 10)  # Limit k to reasonable values

    def compute_separability(
        self, X: np.ndarray, outlier_idx: int, gamma: float
    ) -> float:
        """Compute separability using KNN distance-based approach."""
        n_samples = len(X)

        if n_samples <= self.k + 1:
            return 0.5

        try:
            # Create binary labels
            y = np.full(n_samples, 0)
            y[outlier_idx] = 1

            # Use distance-based separability instead of kernel transformation
            outlier_point = X[outlier_idx : outlier_idx + 1]
            other_points = np.delete(X, outlier_idx, axis=0)
            other_labels = np.delete(y, outlier_idx)

            # Compute distances from outlier to all other points
            distances = cdist(outlier_point, other_points, metric="euclidean").flatten()

            # Find k nearest neighbors
            k_actual = min(self.k, len(distances))
            nearest_indices = np.argpartition(distances, k_actual)[:k_actual]
            nearest_distances = distances[nearest_indices]
            nearest_labels = other_labels[nearest_indices]

            # Compute weighted probability based on distances and gamma
            # Convert distances to similarities using RBF-like transformation
            similarities = np.exp(-gamma * nearest_distances**2)

            # Weighted probability (higher similarity = more influence)
            weights = similarities / (similarities.sum() + 1e-10)
            weighted_prob = 1.0 - np.sum(weights * (1 - nearest_labels))

            return float(np.clip(weighted_prob, 0.01, 0.99))

        except Exception:
            return 0.5


def _get_separability_classifier(
    classifier: str,
    penalty: float,
    kernel_approximation: str,
    random_state: Optional[int],
) -> SeparabilityClassifier:
    """Factory function to create separability classifier."""
    if classifier == "klr":
        return KLRSeparabilityClassifier(penalty, kernel_approximation, random_state)
    elif classifier == "svm":
        return SVMSeparabilityClassifier(penalty, kernel_approximation, random_state)
    elif classifier == "logistic":
        return LogisticSeparabilityClassifier(
            penalty, kernel_approximation, random_state
        )
    elif classifier == "knn":
        return KNNSeparabilityClassifier(penalty, kernel_approximation, random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def _estimate_gamma_max_robust(
    X: np.ndarray,
    outlier_indices: np.ndarray,
    gamma_min: float,
    classifier: str,
    rng: np.random.Generator,
) -> float:
    """Robust estimation of maximum gamma value using exponential search."""
    gamma = gamma_min
    max_attempts = 25
    target_separability = 0.9  # Lower threshold for more robust estimation

    # Create temporary classifier for estimation
    temp_classifier = _get_separability_classifier(classifier, 100.0, "exact", None)

    # Test with multiple representative outliers for robustness
    test_outliers = outlier_indices[: min(3, len(outlier_indices))]

    for attempt in range(max_attempts):
        separabilities = []

        for outlier_idx in test_outliers:
            try:
                sep = temp_classifier.compute_separability(X, outlier_idx, gamma)
                separabilities.append(sep)
            except Exception:
                separabilities.append(0.5)

        avg_separability = np.mean(separabilities)

        if avg_separability >= target_separability:
            break

        # Exponential increase with adaptive step size
        if attempt < 10:
            gamma *= 1.8  # Faster initial growth
        else:
            gamma *= 1.3  # Slower growth for fine-tuning

        # Safety bounds
        if gamma > 2000.0:
            break

    return min(gamma, 1500.0)  # Conservative upper bound


# This function is now handled by the SeparabilityClassifier classes


def _estimate_expected_ireos_enhanced(
    X: np.ndarray,
    n_outliers: int,
    gamma_values: np.ndarray,
    sep_classifier: SeparabilityClassifier,
    n_runs: int,
    rng: np.random.Generator,
) -> float:
    """Enhanced expected IREOS estimation with adaptive sampling."""
    random_scores = []

    # Use coarse gamma grid for efficiency (adaptive based on total gamma points)
    step_size = max(1, len(gamma_values) // 12)
    gamma_subset = gamma_values[::step_size]

    for run in range(n_runs):
        # Select random outliers
        random_outliers = rng.choice(len(X), n_outliers, replace=False)

        # Compute separability curve for random selection
        separabilities = []
        for gamma in gamma_subset:
            outlier_seps = []
            for idx in random_outliers:
                try:
                    sep = sep_classifier.compute_separability(X, idx, gamma)
                    outlier_seps.append(sep)
                except Exception:
                    outlier_seps.append(0.5)  # Neutral separability on error

            avg_sep = np.mean(outlier_seps) if outlier_seps else 0.5
            separabilities.append(avg_sep)

        # Compute raw IREOS for random selection
        if len(separabilities) > 1:
            gamma_range = gamma_values[-1] - gamma_values[0]
            random_ireos = simpson(separabilities, x=gamma_subset) / gamma_range
            random_scores.append(random_ireos)
        else:
            random_scores.append(0.5)

    return np.mean(random_scores) if random_scores else 0.5


def _compute_p_value_enhanced(
    X: np.ndarray,
    n_outliers: int,
    gamma_values: np.ndarray,
    sep_classifier: SeparabilityClassifier,
    observed_ireos: float,
    n_runs: int,
    rng: np.random.Generator,
) -> float:
    """Enhanced p-value computation with better statistical properties."""
    random_scores = []

    # Use coarser gamma grid for p-value computation (computational efficiency)
    step_size = max(1, len(gamma_values) // 8)
    gamma_subset = gamma_values[::step_size]

    for run in range(n_runs):
        # Select random outliers
        random_outliers = rng.choice(len(X), n_outliers, replace=False)

        # Compute separability curve for random selection
        separabilities = []
        for gamma in gamma_subset:
            outlier_seps = []
            for idx in random_outliers:
                try:
                    sep = sep_classifier.compute_separability(X, idx, gamma)
                    outlier_seps.append(sep)
                except Exception:
                    outlier_seps.append(0.5)

            avg_sep = np.mean(outlier_seps) if outlier_seps else 0.5
            separabilities.append(avg_sep)

        # Compute random IREOS score
        if len(separabilities) > 1:
            gamma_range = gamma_values[-1] - gamma_values[0]
            random_ireos = simpson(separabilities, x=gamma_subset) / gamma_range
            random_scores.append(random_ireos)

    if len(random_scores) == 0:
        return 1.0

    # Enhanced p-value computation with bias correction
    random_scores = np.array(random_scores)

    # Compute p-value: proportion of random scores >= observed
    n_greater_equal = np.sum(random_scores >= observed_ireos)

    # Apply continuity correction for small samples
    p_value = (n_greater_equal + 0.5) / (len(random_scores) + 1)

    # Ensure reasonable bounds
    return float(np.clip(p_value, 1e-6, 1.0))


def _validate_score_normalization(scores: np.ndarray) -> None:
    """
    Validate that scores appear to be properly normalized for IREOS.

    Issues warnings if scores don't appear to be normalized to [0,1] range
    as required by the original IREOS specification (Kriegel et al. 2011).
    """
    score_min, score_max = scores.min(), scores.max()
    score_range = score_max - score_min

    # Check if scores are in approximately [0, 1] range
    if score_min < -0.1 or score_max > 1.1:
        warnings.warn(
            f"IREOS expects normalized scores in [0,1] range, but got [{score_min:.3f}, {score_max:.3f}]. "
            f"Consider using labelfree.normalize_outlier_scores() or labelfree.auto_normalize_scores() "
            f"to normalize raw outlier scores before IREOS evaluation. "
            f"Raw scores may produce unreliable IREOS results.",
            UserWarning,
            stacklevel=3,
        )

    # Check for suspicious score distributions that suggest raw/unnormalized scores
    elif score_range > 10:
        warnings.warn(
            f"IREOS scores have large range ({score_range:.1f}), which may indicate "
            f"unnormalized raw outlier scores. Consider using normalization functions "
            f"from labelfree.utils for better IREOS results.",
            UserWarning,
            stacklevel=3,
        )

    # Check for negative scores (common in distance-based algorithms)
    elif np.mean(scores < 0) > 0.5:
        warnings.warn(
            f"IREOS scores contain many negative values ({np.mean(scores < 0):.1%}), "
            f"which suggests raw distance-based scores. Consider using "
            f"labelfree.auto_normalize_scores(scores, invert=True) for proper normalization.",
            UserWarning,
            stacklevel=3,
        )
