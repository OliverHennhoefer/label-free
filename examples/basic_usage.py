"""Basic usage example of label-free metrics."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs

import labelfree


def main():
    # Generate synthetic data
    print("Generating synthetic data...")
    X_normal, _ = make_blobs(n_samples=800, centers=3, n_features=2, random_state=42)
    
    # Add some anomalies
    rng = np.random.default_rng(42)
    X_anomalies = rng.uniform(-8, 8, size=(50, 2))
    X = np.vstack([X_normal, X_anomalies])
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train anomaly detector
    print("\nTraining Isolation Forest...")
    detector = IsolationForest(contamination=0.1, random_state=42)
    detector.fit(X)
    scores = detector.score_samples(X)
    
    # Convert to positive scores (higher = more anomalous)
    scores = -scores
    
    print(f"Anomaly scores range: [{scores.min():.3f}, {scores.max():.3f}]")
    
    # Evaluate with combined EM/MV scores
    print("\n=== EM/MV Scores ===")
    em_score, mv_score = labelfree.emmv_scores(
        scores, X,
        random_state=42,
        scoring_function=lambda x: -detector.score_samples(x)
    )
    print(f"Excess Mass score: {em_score:.4f} (higher is better)")
    print(f"Mass Volume score: {mv_score:.4f} (lower is better)")
    
    # Also show individual curves
    print("\n=== Mass-Volume Curve ===")
    mv_result = labelfree.mass_volume_curve(
        scores, X, 
        n_thresholds=50, 
        random_state=42,
        scoring_function=lambda x: -detector.score_samples(x)
    )
    print(f"MV-AUC: {mv_result['auc']:.4f} (lower is better)")
    
    # Evaluate with IREOS
    print("\n=== IREOS ===")
    ireos_score, p_value = labelfree.ireos(scores, X, random_state=42)
    print(f"IREOS Score: {ireos_score:.4f} (higher is better)")
    print(f"P-value: {p_value:.4f}")
    
    # Evaluate with SIREOS
    print("\n=== SIREOS ===")
    sireos_score = labelfree.sireos(scores, X, similarity='euclidean')
    print(f"SIREOS Score: {sireos_score:.4f} (higher is better)")
    
    # Check ranking stability
    print("\n=== Ranking Stability ===")
    stability = labelfree.ranking_stability(
        detector.score_samples, X, n_subsamples=10, random_state=42
    )
    print(f"Mean Stability: {stability['mean']:.4f} ± {stability['std']:.4f}")
    print(f"Min Stability: {stability['min']:.4f}")
    
    # Check top-k stability
    print("\n=== Top-K Stability ===")
    top_k_stab = labelfree.top_k_stability(
        detector.score_samples, X, k_values=[10, 20, 50], random_state=42
    )
    for k, jaccard in top_k_stab.items():
        print(f"Top-{k} Jaccard: {jaccard:.4f}")


if __name__ == "__main__":
    main()