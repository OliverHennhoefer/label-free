# Label-Free Metrics - Clean Implementation

## Project Structure

```
labelfree/
├── __init__.py
├── mass_volume.py
├── excess_mass.py
├── ireos.py
├── stability.py
├── utils.py
└── tests/
    ├── __init__.py
    ├── test_mass_volume.py
    ├── test_excess_mass.py
    ├── test_ireos.py
    ├── test_stability.py
    └── synthetic_data.py
```

## Core Implementations

### utils.py

```python
"""Utility functions for label-free metrics."""
import numpy as np
from typing import Tuple, Optional


def validate_scores(scores: np.ndarray, name: str = "scores") -> np.ndarray:
    """Ensure scores are valid 1D array."""
    scores = np.asarray(scores)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be 1D array, got shape {scores.shape}")
    if len(scores) == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} contains non-finite values")
    return scores


def validate_data(data: np.ndarray, name: str = "data") -> np.ndarray:
    """Ensure data is valid 2D array."""
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {data.shape}")
    if len(data) == 0:
        raise ValueError(f"{name} cannot be empty")
    return data


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under curve using trapezoidal rule."""
    # Sort by x values for proper integration
    idx = np.argsort(x)
    return float(np.trapz(y[idx], x[idx]))
```

### mass_volume.py

```python
"""Mass-Volume curve implementation."""
import numpy as np
from typing import Tuple, Optional, Dict
from .utils import validate_scores, validate_data, compute_auc


def mass_volume_curve(
    scores: np.ndarray,
    data: np.ndarray,
    n_thresholds: int = 100,
    n_mc_samples: int = 10000,
    random_state: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """
    Compute Mass-Volume curve for anomaly detection evaluation.
    
    The MV curve shows the trade-off between mass (fraction of data captured)
    and volume (fraction of space occupied) at different score thresholds.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores where higher values indicate anomalies.
    data : array-like of shape (n_samples, n_features)
        Original data points corresponding to scores.
    n_thresholds : int, default=100
        Number of thresholds to evaluate.
    n_mc_samples : int, default=10000
        Number of Monte Carlo samples for volume estimation.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    dict with keys:
        - 'mass': Mass values at each threshold
        - 'volume': Volume values at each threshold
        - 'auc': Area under the MV curve (lower is better)
        - 'thresholds': Score thresholds used
    """
    scores = validate_scores(scores)
    data = validate_data(data)
    
    if len(scores) != len(data):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(data)} data points")
    
    rng = np.random.default_rng(random_state)
    
    # Compute thresholds based on score quantiles
    quantiles = np.linspace(0, 100, n_thresholds)
    thresholds = np.percentile(scores, quantiles)
    
    # Generate uniform samples in data bounding box
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    uniform_samples = rng.uniform(data_min, data_max, size=(n_mc_samples, data.shape[1]))
    
    # Compute masses and volumes
    masses = np.zeros(n_thresholds)
    volumes = np.zeros(n_thresholds)
    
    for i, threshold in enumerate(thresholds):
        # Mass: fraction of data with score >= threshold
        masses[i] = (scores >= threshold).mean()
        
        # Volume: estimated by fraction of uniform samples with score >= threshold
        # Note: In practice, you'd need to score the uniform samples with your model
        # For testing, we'll simulate this
        uniform_scores = _simulate_uniform_scores(uniform_samples, data, scores, rng)
        volumes[i] = (uniform_scores >= threshold).mean()
    
    # Compute area under curve
    auc = compute_auc(masses, volumes)
    
    return {
        'mass': masses,
        'volume': volumes,
        'auc': auc,
        'thresholds': thresholds
    }


def _simulate_uniform_scores(uniform_samples, data, scores, rng):
    """Simulate scores for uniform samples based on nearest neighbors."""
    # Simple approach: assign score based on distance to nearest data point
    from scipy.spatial import cKDTree
    tree = cKDTree(data)
    distances, indices = tree.query(uniform_samples, k=1)
    
    # Add noise to avoid exact copies
    base_scores = scores[indices]
    noise = rng.normal(0, 0.1 * scores.std(), size=len(uniform_samples))
    return base_scores + noise
```

### excess_mass.py

```python
"""Excess-Mass curve implementation."""
import numpy as np
from typing import Dict, Optional
from .utils import validate_scores, compute_auc


def excess_mass_curve(
    scores: np.ndarray,
    volume_scores: np.ndarray,
    n_levels: int = 100,
    volume: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    Compute Excess-Mass curve for anomaly detection evaluation.
    
    The Excess-Mass at level t measures how well the scoring function
    captures high-density regions: EM(t) = P(score > s) - t * V(score > s)
    where V is the volume measure.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores on actual data.
    volume_scores : array-like of shape (n_uniform_samples,)
        Anomaly scores on uniform samples (for volume estimation).
    n_levels : int, default=100
        Number of levels to evaluate.
    volume : float, default=1.0
        Total volume of the data space.
        
    Returns
    -------
    dict with keys:
        - 'levels': Level values t
        - 'excess_mass': EM values at each level
        - 'auc': Area under EM curve (higher is better)
        - 'max_em': Maximum excess mass achieved
    """
    scores = validate_scores(scores)
    volume_scores = validate_scores(volume_scores, "volume_scores")
    
    # Generate levels
    levels = np.linspace(0, 100.0 / volume, n_levels)
    
    # Find unique score thresholds from data
    unique_thresholds = np.unique(scores)
    
    # Compute excess mass for each level
    excess_masses = np.zeros(n_levels)
    
    for i, level in enumerate(levels):
        # Find optimal threshold for this level
        max_em = -np.inf
        
        for threshold in unique_thresholds:
            # P(score > threshold) on data
            p_data = (scores > threshold).mean()
            
            # P(score > threshold) on uniform
            p_uniform = (volume_scores > threshold).mean()
            
            # Excess mass at this threshold and level
            em = p_data - level * p_uniform * volume
            max_em = max(max_em, em)
        
        excess_masses[i] = max_em
    
    # Area under curve
    auc = compute_auc(levels, excess_masses)
    
    return {
        'levels': levels,
        'excess_mass': excess_masses,
        'auc': auc,
        'max_em': excess_masses.max()
    }
```

### ireos.py

```python
"""Internal Relative Evaluation of Outlier Solutions (IREOS)."""
import numpy as np
from typing import Tuple, Optional
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from .utils import validate_scores, validate_data


def ireos(
    scores: np.ndarray,
    data: np.ndarray,
    n_splits: int = 5,
    random_state: Optional[int] = None
) -> Tuple[float, float]:
    """
    Compute IREOS (Internal Relative Evaluation of Outlier Solutions).
    
    IREOS measures the separability between high-scoring (anomalous) and
    low-scoring (normal) points using a classifier. No labels required.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores from detector.
    data : array-like of shape (n_samples, n_features)
        Original features corresponding to scores.
    n_splits : int, default=5
        Number of cross-validation splits.
    random_state : int, optional
        Random seed for reproducibility.
        
    Returns
    -------
    ireos_score : float
        Separability score (higher is better), typically in [0.5, 1].
    p_value : float
        Approximate p-value for the score.
    """
    scores = validate_scores(scores)
    data = validate_data(data)
    
    if len(scores) != len(data):
        raise ValueError(f"Length mismatch: {len(scores)} scores vs {len(data)} data points")
    
    # Create binary labels using median split
    median_score = np.median(scores)
    binary_labels = (scores > median_score).astype(int)
    
    # Check for degenerate cases
    n_positive = binary_labels.sum()
    if n_positive == 0 or n_positive == len(binary_labels):
        return 0.5, 1.0  # No separation possible
    
    # Train SVM and evaluate via cross-validation
    svm = SVC(kernel='rbf', random_state=random_state)
    cv_scores = cross_val_score(svm, data, binary_labels, cv=n_splits, scoring='roc_auc')
    
    ireos_score = float(cv_scores.mean())
    
    # Approximate p-value (simplified)
    z_score = (ireos_score - 0.5) / (cv_scores.std() + 1e-10)
    p_value = 2 * (1 - min(0.9999, 0.5 + 0.5 * np.tanh(z_score)))
    
    return ireos_score, p_value


def sireos(
    scores: np.ndarray,
    data: np.ndarray,
    similarity: str = 'euclidean'
) -> float:
    """
    Compute SIREOS (Similarity-based IREOS).
    
    Faster variant using similarity ratios instead of classification.
    
    Parameters
    ----------
    scores : array-like of shape (n_samples,)
        Anomaly scores.
    data : array-like of shape (n_samples, n_features)
        Original features.
    similarity : {'euclidean', 'cosine'}, default='euclidean'
        Similarity metric to use.
        
    Returns
    -------
    sireos_score : float
        Separability score (higher is better).
    """
    scores = validate_scores(scores)
    data = validate_data(data)
    
    # Split based on median
    median_score = np.median(scores)
    high_mask = scores > median_score
    low_mask = ~high_mask
    
    if not any(high_mask) or not any(low_mask):
        return 1.0  # Perfect separation (degenerate case)
    
    # Compute similarity matrix
    if similarity == 'euclidean':
        from scipy.spatial.distance import cdist
        distances = cdist(data, data, metric='euclidean')
        similarities = np.exp(-distances / distances.std())
    else:  # cosine
        normalized = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
        similarities = normalized @ normalized.T
    
    # Compute intra-group and inter-group similarities
    high_high = similarities[high_mask][:, high_mask].mean()
    low_low = similarities[low_mask][:, low_mask].mean()
    high_low = similarities[high_mask][:, low_mask].mean()
    
    # SIREOS score: ratio of intra to inter similarity
    intra_sim = (high_high + low_low) / 2
    inter_sim = high_low
    
    return float(intra_sim / (inter_sim + 1e-10))
```

### stability.py

```python
"""Ranking stability metrics."""
import numpy as np
from typing import Dict, List, Callable
from scipy.stats import kendalltau, spearmanr
from .utils import validate_data


def ranking_stability(
    score_func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    n_subsamples: int = 20,
    subsample_ratio: float = 0.8,
    method: str = 'kendall',
    random_state: Optional[int] = None
) -> Dict[str, float]:
    """
    Measure stability of anomaly rankings under data perturbation.
    
    Parameters
    ----------
    score_func : callable
        Function that takes data and returns anomaly scores.
    data : array-like of shape (n_samples, n_features)
        Dataset to evaluate on.
    n_subsamples : int, default=20
        Number of subsamples to generate.
    subsample_ratio : float, default=0.8
        Fraction of data in each subsample.
    method : {'kendall', 'spearman'}, default='kendall'
        Correlation method to use.
    random_state : int, optional
        Random seed.
        
    Returns
    -------
    dict with keys:
        - 'mean': Mean correlation between rankings
        - 'std': Standard deviation of correlations
        - 'min': Minimum correlation observed
    """
    data = validate_data(data)
    rng = np.random.default_rng(random_state)
    
    n_samples = len(data)
    subsample_size = int(n_samples * subsample_ratio)
    
    # Generate scores for each subsample
    all_scores = []
    all_indices = []
    
    for _ in range(n_subsamples):
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        scores = score_func(data[indices])
        
        all_scores.append(scores)
        all_indices.append(indices)
    
    # Compute pairwise correlations
    correlations = []
    
    for i in range(n_subsamples):
        for j in range(i + 1, n_subsamples):
            # Find common indices
            common = np.intersect1d(all_indices[i], all_indices[j])
            
            if len(common) < 10:  # Skip if too little overlap
                continue
            
            # Get ranks for common points
            pos_i = np.searchsorted(all_indices[i], common)
            pos_j = np.searchsorted(all_indices[j], common)
            
            scores_i = all_scores[i][pos_i]
            scores_j = all_scores[j][pos_j]
            
            # Compute correlation
            if method == 'kendall':
                corr, _ = kendalltau(scores_i, scores_j)
            else:
                corr, _ = spearmanr(scores_i, scores_j)
            
            if not np.isnan(corr):
                correlations.append(corr)
    
    correlations = np.array(correlations)
    
    return {
        'mean': float(correlations.mean()),
        'std': float(correlations.std()),
        'min': float(correlations.min())
    }


def top_k_stability(
    score_func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    k_values: List[int] = [10, 50, 100],
    n_subsamples: int = 20,
    subsample_ratio: float = 0.8,
    random_state: Optional[int] = None
) -> Dict[int, float]:
    """
    Measure stability of top-k anomaly detection.
    
    Returns
    -------
    dict mapping k to average Jaccard similarity of top-k sets.
    """
    data = validate_data(data)
    rng = np.random.default_rng(random_state)
    
    n_samples = len(data)
    subsample_size = int(n_samples * subsample_ratio)
    
    # Collect top-k for each subsample
    top_k_sets = {k: [] for k in k_values}
    
    for _ in range(n_subsamples):
        indices = rng.choice(n_samples, size=subsample_size, replace=False)
        scores = score_func(data[indices])
        
        # Get top-k indices (highest scores)
        top_indices = indices[np.argsort(scores)[::-1]]
        
        for k in k_values:
            if k <= len(top_indices):
                top_k_sets[k].append(set(top_indices[:k]))
    
    # Compute Jaccard similarities
    results = {}
    
    for k in k_values:
        if not top_k_sets[k]:
            continue
            
        similarities = []
        sets = top_k_sets[k]
        
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                intersection = len(sets[i] & sets[j])
                union = len(sets[i] | sets[j])
                similarities.append(intersection / union if union > 0 else 0)
        
        results[k] = float(np.mean(similarities)) if similarities else 0.0
    
    return results
```

## Test Suite

### tests/synthetic_data.py

```python
"""Generate synthetic data for testing label-free metrics."""
import numpy as np
from typing import Tuple, Optional


def make_blobs_with_anomalies(
    n_samples: int = 1000,
    n_features: int = 2,
    n_anomalies: int = 50,
    centers: int = 3,
    anomaly_factor: float = 3.0,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate blob data with anomalies far from centers.
    
    Returns
    -------
    X : array of shape (n_samples + n_anomalies, n_features)
        Data points.
    y : array of shape (n_samples + n_anomalies,)
        Labels where 0=normal, 1=anomaly.
    """
    from sklearn.datasets import make_blobs
    
    # Generate normal data
    X_normal, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        random_state=random_state
    )
    
    # Generate anomalies far from normal data
    rng = np.random.default_rng(random_state)
    
    # Find data range
    data_min = X_normal.min(axis=0)
    data_max = X_normal.max(axis=0)
    data_range = data_max - data_min
    
    # Place anomalies outside normal range
    X_anomalies = rng.uniform(
        data_min - anomaly_factor * data_range,
        data_max + anomaly_factor * data_range,
        size=(n_anomalies, n_features)
    )
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])
    
    return X, y


def make_moons_with_noise(
    n_samples: int = 1000,
    noise_level: float = 0.3,
    n_anomalies: int = 50,
    random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate two moons data with uniform noise as anomalies.
    
    Returns
    -------
    X : array of shape (n_samples + n_anomalies, 2)
        Data points.
    y : array of shape (n_samples + n_anomalies,)
        Labels where 0=normal, 1=anomaly.
    """
    from sklearn.datasets import make_moons
    
    # Generate moons
    X_normal, _ = make_moons(
        n_samples=n_samples,
        noise=noise_level,
        random_state=random_state
    )
    
    # Generate uniform anomalies
    rng = np.random.default_rng(random_state)
    X_anomalies = rng.uniform(-3, 3, size=(n_anomalies, 2))
    
    X = np.vstack([X_normal, X_anomalies])
    y = np.hstack([np.zeros(n_samples), np.ones(n_anomalies)])
    
    return X, y


def make_anomaly_scores(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'distance',
    noise_level: float = 0.1,
    random_state: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic anomaly scores for testing.
    
    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        Data points.
    y : array of shape (n_samples,)
        True labels (0=normal, 1=anomaly).
    method : {'distance', 'random', 'perfect'}, default='distance'
        How to generate scores.
    noise_level : float, default=0.1
        Amount of noise to add.
        
    Returns
    -------
    scores : array of shape (n_samples,)
        Anomaly scores (higher = more anomalous).
    """
    rng = np.random.default_rng(random_state)
    
    if method == 'distance':
        # Score based on distance from center of normal points
        normal_center = X[y == 0].mean(axis=0)
        scores = np.linalg.norm(X - normal_center, axis=1)
        
    elif method == 'random':
        # Random scores
        scores = rng.random(len(X))
        
    elif method == 'perfect':
        # Perfect scores based on labels
        scores = y.astype(float)
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Add noise
    if noise_level > 0:
        scores += rng.normal(0, noise_level * scores.std(), size=len(scores))
    
    return scores
```

### tests/test_mass_volume.py

```python
"""Tests for Mass-Volume curve."""
import numpy as np
import pytest
from label_free_metrics.mass_volume import mass_volume_curve
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestMassVolume:
    """Test suite for Mass-Volume curve."""
    
    def test_basic_functionality(self):
        """Test basic MV curve computation."""
        # Generate simple data
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        # Compute MV curve
        result = mass_volume_curve(scores, X, n_thresholds=50, n_mc_samples=1000)
        
        # Check output structure
        assert set(result.keys()) == {'mass', 'volume', 'auc', 'thresholds'}
        assert len(result['mass']) == 50
        assert len(result['volume']) == 50
        assert len(result['thresholds']) == 50
        
        # Check value ranges
        assert np.all((result['mass'] >= 0) & (result['mass'] <= 1))
        assert np.all((result['volume'] >= 0) & (result['volume'] <= 1))
        assert 0 <= result['auc'] <= 1
        
        # Mass and volume should be monotonic
        assert np.all(np.diff(result['mass']) <= 0)  # Decreasing
        assert np.all(np.diff(result['volume']) <= 0)  # Decreasing
    
    def test_perfect_detector(self):
        """Test MV curve for perfect anomaly detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method='perfect', noise_level=0)
        
        result = mass_volume_curve(scores, X, n_thresholds=100)
        
        # Perfect detector should have very low AUC
        assert result['auc'] < 0.2
    
    def test_random_detector(self):
        """Test MV curve for random detector."""
        X, y = make_blobs_with_anomalies(n_samples=500, n_anomalies=50, random_state=42)
        scores = make_anomaly_scores(X, y, method='random')
        
        result = mass_volume_curve(scores, X, n_thresholds=100)
        
        # Random detector should have AUC close to 0.5
        assert 0.4 <= result['auc'] <= 0.6
    
    def test_input_validation(self):
        """Test input validation."""
        X = np.random.randn(100, 2)
        scores = np.random.randn(100)
        
        # Mismatched lengths
        with pytest.raises(ValueError):
            mass_volume_curve(scores[:50], X)
        
        # Empty inputs
        with pytest.raises(ValueError):
            mass_volume_curve([], X)
        
        # Invalid dimensions
        with pytest.raises(ValueError):
            mass_volume_curve(scores.reshape(-1, 1), X)
    
    def test_reproducibility(self):
        """Test that results are reproducible with fixed seed."""
        X, y = make_blobs_with_anomalies(random_state=42)
        scores = make_anomaly_scores(X, y, random_state=42)
        
        result1 = mass_volume_curve(scores, X, random_state=42)
        result2 = mass_volume_curve(scores, X, random_state=42)
        
        np.testing.assert_array_equal(result1['mass'], result2['mass'])
        np.testing.assert_array_equal(result1['volume'], result2['volume'])
        assert result1['auc'] == result2['auc']
```

### tests/test_excess_mass.py

```python
"""Tests for Excess-Mass curve."""
import numpy as np
import pytest
from label_free_metrics.excess_mass import excess_mass_curve
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestExcessMass:
    """Test suite for Excess-Mass curve."""
    
    def test_basic_functionality(self):
        """Test basic EM curve computation."""
        # Generate data and scores
        X, y = make_blobs_with_anomalies(n_samples=300, n_anomalies=30, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        # Generate volume scores (scores on uniform samples)
        volume_scores = np.random.randn(1000)
        
        # Compute EM curve
        result = excess_mass_curve(scores, volume_scores, n_levels=50)
        
        # Check output
        assert set(result.keys()) == {'levels', 'excess_mass', 'auc', 'max_em'}
        assert len(result['levels']) == 50
        assert len(result['excess_mass']) == 50
        assert isinstance(result['auc'], float)
        assert isinstance(result['max_em'], float)
        
        # Check that max_em is actually the maximum
        assert result['max_em'] == pytest.approx(result['excess_mass'].max())
    
    def test_edge_cases(self):
        """Test edge cases."""
        # All scores identical
        scores = np.ones(100)
        volume_scores = np.ones(100)
        
        result = excess_mass_curve(scores, volume_scores)
        
        # Should still run without errors
        assert result['max_em'] >= 0
        
        # Empty volume scores
        with pytest.raises(ValueError):
            excess_mass_curve(scores, [])
```

### tests/test_ireos.py

```python
"""Tests for IREOS."""
import numpy as np
import pytest
from label_free_metrics.ireos import ireos, sireos
from .synthetic_data import make_blobs_with_anomalies, make_anomaly_scores


class TestIREOS:
    """Test suite for IREOS metrics."""
    
    def test_ireos_basic(self):
        """Test basic IREOS functionality."""
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        ireos_score, p_value = ireos(scores, X, n_splits=3, random_state=42)
        
        # Check ranges
        assert 0.5 <= ireos_score <= 1.0
        assert 0.0 <= p_value <= 1.0
        
        # Good detector should have high IREOS
        assert ireos_score > 0.7
        assert p_value < 0.05
    
    def test_ireos_random_scores(self):
        """Test IREOS with random scores."""
        X = np.random.randn(200, 5)
        scores = np.random.randn(200)
        
        ireos_score, p_value = ireos(scores, X, random_state=42)
        
        # Random scores should give IREOS near 0.5
        assert 0.45 <= ireos_score <= 0.55
        assert p_value > 0.1
    
    def test_sireos(self):
        """Test SIREOS functionality."""
        X, y = make_blobs_with_anomalies(n_samples=200, n_anomalies=20, random_state=42)
        scores = make_anomaly_scores(X, y, method='distance', random_state=42)
        
        # Test Euclidean
        sireos_euclidean = sireos(scores, X, similarity='euclidean')
        assert sireos_euclidean > 1.0  # Good separation
        
        # Test cosine
        sireos_cosine = sireos(scores, X, similarity='cosine')
        assert sireos_cosine > 1.0
    
    def test_degenerate_cases(self):
        """Test degenerate cases."""
        X = np.random.randn(100, 2)
        
        # All scores identical
        scores = np.ones(100)
        ireos_score, p_value = ireos(scores, X)
        assert ireos_score == 0.5
        assert p_value == 1.0
        
        # SIREOS with perfect separation
        scores = np.hstack([np.zeros(50), np.ones(50)])
        sireos_score = sireos(scores, X[:100])
        assert sireos_score > 1.0
```

### tests/test_stability.py

```python
"""Tests for ranking stability metrics."""
import numpy as np
import pytest
from label_free_metrics.stability import ranking_stability, top_k_stability
from .synthetic_data import make_blobs_with_anomalies


class TestStability:
    """Test suite for stability metrics."""
    
    def test_ranking_stability(self):
        """Test ranking stability measurement."""
        X, y = make_blobs_with_anomalies(n_samples=500, random_state=42)
        
        # Simple scoring function
        def score_func(data):
            center = data.mean(axis=0)
            return np.linalg.norm(data - center, axis=1)
        
        result = ranking_stability(
            score_func, X, n_subsamples=10, subsample_ratio=0.8, random_state=42
        )
        
        # Check output
        assert set(result.keys()) == {'mean', 'std', 'min'}
        assert 0 <= result['mean'] <= 1
        assert result['std'] >= 0
        assert result['min'] <= result['mean']
        
        # Stable scoring should have high correlation
        assert result['mean'] > 0.7
    
    def test_top_k_stability(self):
        """Test top-k stability measurement."""
        X, y = make_blobs_with_anomalies(n_samples=500, random_state=42)
        
        def score_func(data):
            center = data.mean(axis=0)
            return np.linalg.norm(data - center, axis=1)
        
        result = top_k_stability(
            score_func, X, k_values=[10, 20, 50], n_subsamples=10, random_state=42
        )
        
        # Check output
        assert set(result.keys()) == {10, 20, 50}
        
        # Jaccard similarity should be in [0, 1]
        for k, similarity in result.items():
            assert 0 <= similarity <= 1
        
        # Smaller k should be more stable
        assert result[10] >= result[50]
    
    def test_unstable_scoring(self):
        """Test with unstable scoring function."""
        X = np.random.randn(200, 5)
        
        # Random scoring function (unstable)
        def score_func(data):
            return np.random.randn(len(data))
        
        result = ranking_stability(score_func, X, n_subsamples=10)
        
        # Should have very low stability
        assert result['mean'] < 0.1
        assert result['min'] < 0.1
```

## Usage Example

```python
import numpy as np
from label_free_metrics import mass_volume_curve, ireos, ranking_stability
from sklearn.ensemble import IsolationForest

# Generate data
X_train = np.random.randn(1000, 2)
X_test = np.vstack([
    np.random.randn(950, 2),  # Normal
    5 + np.random.randn(50, 2)  # Anomalies
])

# Train detector
detector = IsolationForest(contamination=0.05).fit(X_train)
scores = detector.score_samples(X_test)

# Evaluate with Mass-Volume curve
mv_result = mass_volume_curve(scores, X_test)
print(f"MV-AUC: {mv_result['auc']:.3f}")

# Evaluate with IREOS
ireos_score, p_value = ireos(scores, X_test)
print(f"IREOS: {ireos_score:.3f} (p={p_value:.3f})")

# Check stability
stability = ranking_stability(detector.score_samples, X_test)
print(f"Ranking stability: {stability['mean']:.3f} ± {stability['std']:.3f}")
```