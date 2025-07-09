"""Compare multiple anomaly detectors using label-free metrics."""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs
import pandas as pd

import labelfree


def evaluate_detector(name, detector, X, fit_predict=False):
    """Evaluate a single detector using all label-free metrics."""
    print(f"\nEvaluating {name}...")

    # Get scores
    if fit_predict:
        # For LOF, we need to use fit_predict and convert to scores
        labels = detector.fit_predict(X)
        scores = -detector.negative_outlier_factor_  # Higher = more anomalous
    else:
        detector.fit(X)
        scores = -detector.score_samples(X)  # Higher = more anomalous

    results = {}

    # Mass-Volume curve
    # Note: For LOF, we can't provide scoring_function, so simulation is used
    if not fit_predict:
        mv_result = labelfree.mass_volume_curve(
            scores,
            X,
            n_thresholds=50,
            random_state=42,
            scoring_function=lambda x: -detector.score_samples(x),
        )
    else:
        mv_result = labelfree.mass_volume_curve(
            scores, X, n_thresholds=50, random_state=42
        )
    results["MV_AUC"] = mv_result["auc"]

    # IREOS
    ireos_score, p_value = labelfree.ireos(scores, X, random_state=42)
    results["IREOS"] = ireos_score
    results["IREOS_pvalue"] = p_value

    # SIREOS
    sireos_score = labelfree.sireos(scores, X, similarity="euclidean")
    results["SIREOS"] = sireos_score

    # Ranking stability
    if not fit_predict:
        # Can only do stability for detectors with score_samples
        stability = labelfree.ranking_stability(
            detector.score_samples, X, n_subsamples=10, random_state=42
        )
        results["Stability_Mean"] = stability["mean"]
        results["Stability_Std"] = stability["std"]

        # Top-k stability
        top_k_stab = labelfree.top_k_stability(
            detector.score_samples, X, k_values=[10, 20], random_state=42
        )
        results["Top10_Jaccard"] = top_k_stab.get(10, np.nan)
        results["Top20_Jaccard"] = top_k_stab.get(20, np.nan)
    else:
        results["Stability_Mean"] = np.nan
        results["Stability_Std"] = np.nan
        results["Top10_Jaccard"] = np.nan
        results["Top20_Jaccard"] = np.nan

    return results


def main():
    # Generate synthetic dataset
    print("Generating synthetic dataset...")
    X_normal, _ = make_blobs(n_samples=800, centers=4, n_features=3, random_state=42)

    # Add anomalies
    rng = np.random.default_rng(42)
    X_anomalies = rng.uniform(-10, 10, size=(80, 3))
    X = np.vstack([X_normal, X_anomalies])

    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Define detectors
    detectors = [
        (
            "Isolation Forest",
            IsolationForest(contamination=0.1, random_state=42),
            False,
        ),
        ("One-Class SVM", OneClassSVM(nu=0.1), False),
        ("LOF", LocalOutlierFactor(n_neighbors=20, contamination=0.1), True),
    ]

    # Evaluate all detectors
    all_results = {}

    for name, detector, fit_predict in detectors:
        try:
            results = evaluate_detector(name, detector, X, fit_predict)
            all_results[name] = results
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            continue

    # Create comparison table
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 80)

    df = pd.DataFrame(all_results).T

    # Round for better display
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)

    print(df.to_string())

    # Interpretation
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print("MV_AUC: Lower is better (area under Mass-Volume curve)")
    print("IREOS: Higher is better (separability score)")
    print("SIREOS: Higher is better (similarity-based separability)")
    print("Stability_Mean: Higher is better (ranking consistency)")
    print("Top10/20_Jaccard: Higher is better (top-k consistency)")

    # Rank detectors
    print("\n" + "=" * 80)
    print("RANKING (where applicable)")
    print("=" * 80)

    # For metrics where higher is better
    for metric in ["IREOS", "SIREOS", "Stability_Mean", "Top10_Jaccard"]:
        if metric in df.columns and not df[metric].isna().all():
            ranked = df[metric].dropna().sort_values(ascending=False)
            print(f"{metric}: {' > '.join(ranked.index)}")

    # For MV_AUC where lower is better
    if "MV_AUC" in df.columns:
        ranked = df["MV_AUC"].dropna().sort_values(ascending=True)
        print(f"MV_AUC: {' > '.join(ranked.index)} (lower is better)")


if __name__ == "__main__":
    main()
