"""Visualization example for label-free metrics."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

import labelfree


def plot_mass_volume_curve(scores, data, title="Mass-Volume Curve"):
    """Plot the Mass-Volume curve."""
    result = labelfree.mass_volume_curve(scores, data, n_thresholds=100, random_state=42)
    
    plt.figure(figsize=(8, 6))
    plt.plot(result['mass'], result['volume'], 'b-', linewidth=2, label=f'MV Curve (AUC={result["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Detector')
    plt.xlabel('Mass (Fraction of Data)')
    plt.ylabel('Volume (Fraction of Space)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_excess_mass_curve(scores, volume_scores, title="Excess-Mass Curve"):
    """Plot the Excess-Mass curve."""
    result = labelfree.excess_mass_curve(scores, volume_scores, n_levels=100)
    
    plt.figure(figsize=(8, 6))
    plt.plot(result['levels'], result['excess_mass'], 'g-', linewidth=2, 
             label=f'EM Curve (AUC={result["auc"]:.3f})')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    plt.xlabel('Level t')
    plt.ylabel('Excess Mass')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_stability_comparison(detectors, data):
    """Compare ranking stability across detectors."""
    results = []
    
    for name, detector in detectors:
        detector.fit(data)
        stability = labelfree.ranking_stability(
            detector.score_samples, data, n_subsamples=15, random_state=42
        )
        results.append({
            'Detector': name,
            'Mean Stability': stability['mean'],
            'Std Stability': stability['std']
        })
    
    # Create bar plot
    names = [r['Detector'] for r in results]
    means = [r['Mean Stability'] for r in results]
    stds = [r['Std Stability'] for r in results]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=5, alpha=0.7)
    plt.ylabel('Ranking Stability')
    plt.title('Ranking Stability Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_data_and_scores(data, scores, title="Data with Anomaly Scores"):
    """Plot 2D data colored by anomaly scores."""
    if data.shape[1] != 2:
        print("Can only plot 2D data")
        return
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data[:, 0], data[:, 1], c=scores, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Anomaly Score')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Generate 2D data for visualization
    print("Generating 2D dataset for visualization...")
    X_normal, _ = make_blobs(n_samples=500, centers=3, n_features=2, 
                           cluster_std=1.5, random_state=42)
    
    # Add anomalies
    rng = np.random.default_rng(42)
    X_anomalies = rng.uniform(-10, 10, size=(50, 2))
    X = np.vstack([X_normal, X_anomalies])
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train detector
    detector = IsolationForest(contamination=0.1, random_state=42)
    detector.fit(X)
    scores = -detector.score_samples(X)  # Higher = more anomalous
    
    # Plot data with scores
    plot_data_and_scores(X, scores, "Dataset with Isolation Forest Scores")
    
    # Plot Mass-Volume curve
    plot_mass_volume_curve(scores, X, "Mass-Volume Curve - Isolation Forest")
    
    # Generate volume scores for Excess-Mass curve
    # Simulate uniform sampling in the data space
    data_min, data_max = X.min(axis=0), X.max(axis=0)
    uniform_samples = rng.uniform(data_min, data_max, size=(2000, 2))
    volume_scores = -detector.score_samples(uniform_samples)
    
    # Plot Excess-Mass curve
    plot_excess_mass_curve(scores, volume_scores, "Excess-Mass Curve - Isolation Forest")
    
    # Compare stability across different detectors
    from sklearn.svm import OneClassSVM
    
    detectors = [
        ("Isolation Forest", IsolationForest(contamination=0.1, random_state=42)),
        ("One-Class SVM (RBF)", OneClassSVM(nu=0.1, kernel='rbf')),
        ("One-Class SVM (Linear)", OneClassSVM(nu=0.1, kernel='linear')),
    ]
    
    plot_stability_comparison(detectors, X)
    
    # Print IREOS comparison
    print("\n" + "="*50)
    print("IREOS COMPARISON")
    print("="*50)
    
    for name, det in detectors:
        det.fit(X)
        det_scores = -det.score_samples(X)
        ireos_score, p_value = labelfree.ireos(det_scores, X, random_state=42)
        sireos_score = labelfree.sireos(det_scores, X)
        
        print(f"{name}:")
        print(f"  IREOS: {ireos_score:.4f} (p={p_value:.4f})")
        print(f"  SIREOS: {sireos_score:.4f}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing visualization dependencies: {e}")
        print("Install with: pip install matplotlib seaborn")
    except Exception as e:
        print(f"Error during visualization: {e}")