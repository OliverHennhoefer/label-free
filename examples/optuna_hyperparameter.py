import numpy as np
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.datasets import make_blobs

import labelfree


def generate_dataset(n_samples=1000, n_features=5, contamination=0.1, random_state=42):
    """Generate synthetic dataset with anomalies."""
    n_anomalies = int(n_samples * contamination)
    n_normal = n_samples - n_anomalies
    
    # Generate normal data
    X_normal, _ = make_blobs(
        n_samples=n_normal,
        centers=3,
        n_features=n_features,
        cluster_std=1.0,
        random_state=random_state
    )
    
    # Generate anomalies
    rng = np.random.default_rng(random_state)
    data_range = X_normal.max(axis=0) - X_normal.min(axis=0)
    X_anomalies = rng.uniform(
        X_normal.min(axis=0) - 2 * data_range,
        X_normal.max(axis=0) + 2 * data_range,
        size=(n_anomalies, n_features)
    )
    
    return np.vstack([X_normal, X_anomalies])


def objective_isolation_forest(trial, X):
    """Objective function for Isolation Forest hyperparameter tuning."""
    # Hyperparameter space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
    max_features = trial.suggest_float("max_features", 0.5, 1.0)
    
    # Train model
    detector = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        contamination=0.000001,
        max_features=max_features,
        random_state=42
    )
    
    detector.fit(X)
    scores = -detector.score_samples(X)
    
    # Evaluate using label-free metrics
    # IREOS - higher is better
    ireos_score, _ = labelfree.ireos(scores, X, random_state=42)
    
    # Mass-Volume AUC - lower is better (so we negate it)
    mv_result = labelfree.mass_volume_curve(scores, X, random_state=42)
    mv_auc = -mv_result["auc"]  # Negate for maximization
    
    # SIREOS - higher is better
    sireos_score = labelfree.sireos(scores, X)
    
    # Ranking stability - higher is better
    stability = labelfree.ranking_stability(
        detector.score_samples, X, n_subsamples=10, random_state=42
    )
    stability_score = stability["mean"]
    
    # Combined objective (weighted sum)
    objective = (
        0.4 * ireos_score +
        0.3 * mv_auc +
        0.2 * sireos_score +
        0.1 * stability_score
    )
    
    return objective


def objective_one_class_svm(trial, X):
    """Objective function for One-Class SVM hyperparameter tuning."""
    # Hyperparameter space
    nu = trial.suggest_float("nu", 0.01, 0.3)
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
    if gamma == "scale":
        gamma_val = trial.suggest_float("gamma_val", 0.001, 10.0, log=True)
    else:
        gamma_val = gamma
    
    # Train model
    detector = OneClassSVM(
        nu=nu,
        kernel="rbf",
        gamma=gamma_val
    )
    
    detector.fit(X)
    scores = -detector.score_samples(X)
    
    # Evaluate using label-free metrics
    ireos_score, _ = labelfree.ireos(scores, X, random_state=42)
    
    # Mass-Volume AUC - lower is better (so we negate it)
    mv_result = labelfree.mass_volume_curve(scores, X, random_state=42)
    mv_auc = -mv_result["auc"]
    
    # SIREOS - higher is better
    sireos_score = labelfree.sireos(scores, X)
    
    # Ranking stability - higher is better
    stability = labelfree.ranking_stability(
        detector.score_samples, X, n_subsamples=10, random_state=42
    )
    stability_score = stability["mean"]
    
    # Combined objective (weighted sum)
    objective = (
        0.4 * ireos_score +
        0.3 * mv_auc +
        0.2 * sireos_score +
        0.1 * stability_score
    )
    
    return objective


def evaluate_best_model(name, detector, X):
    """Evaluate the best model using all metrics."""
    detector.fit(X)
    scores = -detector.score_samples(X)
    
    # All metrics
    ireos_score, p_value = labelfree.ireos(scores, X, random_state=42)
    sireos_score = labelfree.sireos(scores, X)
    mv_result = labelfree.mass_volume_curve(scores, X, random_state=42)
    stability = labelfree.ranking_stability(
        detector.score_samples, X, n_subsamples=15, random_state=42
    )
    top_k_stab = labelfree.top_k_stability(
        detector.score_samples, X, k_values=[10, 20], random_state=42
    )
    
    return {
        "Model": name,
        "IREOS": ireos_score,
        "IREOS_pvalue": p_value,
        "SIREOS": sireos_score,
        "MV_AUC": mv_result["auc"],
        "Stability_Mean": stability["mean"],
        "Stability_Std": stability["std"],
        "Top10_Jaccard": top_k_stab.get(10, np.nan),
        "Top20_Jaccard": top_k_stab.get(20, np.nan)
    }


def main():
    """Main hyperparameter tuning pipeline."""
    print("=" * 80)
    print("UNSUPERVISED HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 80)
    
    # Generate dataset
    print("Generating dataset...")
    X = generate_dataset(n_samples=800, n_features=4, contamination=0.1, random_state=42)
    print(f"Dataset shape: {X.shape}")

    # Tune Isolation Forest
    print("\nTuning Isolation Forest...")
    study_if = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study_if.optimize(
        lambda trial: objective_isolation_forest(trial, X),
        n_trials=50,
        show_progress_bar=True
    )
    
    print(f"Best Isolation Forest score: {study_if.best_value:.4f}")
    print(f"Best parameters: {study_if.best_params}")
    
    # Create best Isolation Forest
    best_if = IsolationForest(
        **study_if.best_params,
        random_state=42
    )
    
    # Tune One-Class SVM
    print("\nTuning One-Class SVM...")
    study_svm = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    
    study_svm.optimize(
        lambda trial: objective_one_class_svm(trial, X),
        n_trials=50,
        show_progress_bar=True
    )
    
    print(f"Best One-Class SVM score: {study_svm.best_value:.4f}")
    print(f"Best parameters: {study_svm.best_params}")
    
    # Create best One-Class SVM
    best_svm_params = study_svm.best_params.copy()
    if "gamma_val" in best_svm_params:
        best_svm_params["gamma"] = best_svm_params.pop("gamma_val")
    
    best_svm = OneClassSVM(
        **best_svm_params,
        kernel="rbf"
    )
    
    # Evaluate best models
    print("\n" + "=" * 80)
    print("EVALUATION OF BEST MODELS")
    print("=" * 80)
    
    if_results = evaluate_best_model("Isolation Forest (Tuned)", best_if, X)
    svm_results = evaluate_best_model("One-Class SVM (Tuned)", best_svm, X)
    
    # Baseline comparison
    baseline_if = IsolationForest(contamination=0.1, random_state=42)
    baseline_svm = OneClassSVM(nu=0.1)
    
    baseline_if_results = evaluate_best_model("Isolation Forest (Default)", baseline_if, X)
    baseline_svm_results = evaluate_best_model("One-Class SVM (Default)", baseline_svm, X)
    
    # Create comparison table
    import pandas as pd
    
    all_results = [if_results, svm_results, baseline_if_results, baseline_svm_results]
    df = pd.DataFrame(all_results)
    
    # Round numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(4)
    
    print(df.to_string(index=False))
    
    # Interpretation
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING SUMMARY")
    print("=" * 80)
    
    print(f"Isolation Forest improvement:")
    print(f"  IREOS: {if_results['IREOS']:.4f} vs {baseline_if_results['IREOS']:.4f} (diff={if_results['IREOS'] - baseline_if_results['IREOS']:.4f})")
    print(f"  MV_AUC: {if_results['MV_AUC']:.4f} vs {baseline_if_results['MV_AUC']:.4f} (diff={if_results['MV_AUC'] - baseline_if_results['MV_AUC']:.4f})")
    print(f"  Stability: {if_results['Stability_Mean']:.4f} vs {baseline_if_results['Stability_Mean']:.4f} (diff={if_results['Stability_Mean'] - baseline_if_results['Stability_Mean']:.4f})")
    
    print(f"\nOne-Class SVM improvement:")
    print(f"  IREOS: {svm_results['IREOS']:.4f} vs {baseline_svm_results['IREOS']:.4f} (diff={svm_results['IREOS'] - baseline_svm_results['IREOS']:.4f})")
    print(f"  MV_AUC: {svm_results['MV_AUC']:.4f} vs {baseline_svm_results['MV_AUC']:.4f} (diff={svm_results['MV_AUC'] - baseline_svm_results['MV_AUC']:.4f})")
    print(f"  Stability: {svm_results['Stability_Mean']:.4f} vs {baseline_svm_results['Stability_Mean']:.4f} (diff={svm_results['Stability_Mean'] - baseline_svm_results['Stability_Mean']:.4f})")
    
    # Best overall model
    tuned_models = [if_results, svm_results]
    best_model = max(tuned_models, key=lambda x: x['IREOS'])
    
    print(f"\nBest overall model: {best_model['Model']}")
    print(f"  IREOS: {best_model['IREOS']:.4f}")
    print(f"  MV_AUC: {best_model['MV_AUC']:.4f}")
    print(f"  Stability: {best_model['Stability_Mean']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install optuna pandas")
    except Exception as e:
        print(f"Error during hyperparameter tuning: {e}")