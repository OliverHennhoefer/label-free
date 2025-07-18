#!/usr/bin/env python3
"""
Optuna hyperparameter optimization for IsolationForest and OneClassSVM
using label-free metrics for evaluation.

Usage:
    python optuna_hyperparams.py --algorithm isolation_forest --data_file data.csv
    python optuna_hyperparams.py --algorithm one_class_svm --data_file data.csv
"""

import argparse
import pandas as pd
import optuna
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import labelfree


def create_isolation_forest_objective(X_train, X_val):
    """Create objective function for IsolationForest hyperparameter tuning."""

    def objective(trial):
        # Suggest hyperparameters
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_samples": trial.suggest_uniform("max_samples", 0.1, 1.0),
            "max_features": trial.suggest_uniform("max_features", 0.1, 1.0),
            "contamination": trial.suggest_uniform("contamination", 0.01, 0.3),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": 42,
        }

        # Train model
        model = IsolationForest(**params)
        model.fit(X_train)

        # Get anomaly scores
        scores = -model.score_samples(X_val)  # Higher = more anomalous

        # Compute label-free metrics
        mv_result = labelfree.mass_volume_auc(
            scores, X_val, n_thresholds=50, random_state=42
        )

        ireos_score, _ = labelfree.ireos(scores, X_val, random_state=42)

        stability_result = labelfree.ranking_stability(
            lambda x: -IsolationForest(**params).fit(x).score_samples(x),
            X_val,
            n_subsamples=10,
            subsample_size=0.8,
            random_state=42,
        )

        # Combine metrics (lower MV_AUC is better, higher IREOS and stability is better)
        composite_score = (
            -mv_result["auc"]  # Negative because lower is better
            + ireos_score
            + stability_result["mean_correlation"]
        ) / 3.0

        return composite_score

    return objective


def create_one_class_svm_objective(X_train, X_val):
    """Create objective function for OneClassSVM hyperparameter tuning."""

    def objective(trial):
        # Suggest hyperparameters
        kernel = trial.suggest_categorical(
            "kernel", ["rbf", "linear", "poly", "sigmoid"]
        )

        params = {
            "kernel": kernel,
            "nu": trial.suggest_uniform("nu", 0.01, 0.5),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            "random_state": 42,
        }

        # Add kernel-specific parameters
        if kernel == "rbf":
            if params["gamma"] == "scale":
                params["gamma"] = trial.suggest_loguniform("gamma_value", 1e-6, 1e-1)
        elif kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)
            params["coef0"] = trial.suggest_uniform("coef0", 0.0, 1.0)
        elif kernel == "sigmoid":
            params["coef0"] = trial.suggest_uniform("coef0", 0.0, 1.0)

        # Train model
        model = OneClassSVM(**params)
        model.fit(X_train)

        # Get anomaly scores
        scores = -model.score_samples(X_val)  # Higher = more anomalous

        # Compute label-free metrics
        mv_result = labelfree.mass_volume_auc(
            scores, X_val, n_thresholds=50, random_state=42
        )

        ireos_score, _ = labelfree.ireos(scores, X_val, random_state=42)

        stability_result = labelfree.ranking_stability(
            lambda x: -OneClassSVM(**params).fit(x).score_samples(x),
            X_val,
            n_subsamples=10,
            subsample_size=0.8,
            random_state=42,
        )

        # Combine metrics (lower MV_AUC is better, higher IREOS and stability is better)
        composite_score = (
            -mv_result["auc"]  # Negative because lower is better
            + ireos_score
            + stability_result["mean_correlation"]
        ) / 3.0

        return composite_score

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization")
    parser.add_argument(
        "--algorithm",
        required=True,
        choices=["isolation_forest", "one_class_svm"],
        help="Algorithm to optimize",
    )
    parser.add_argument("--data_file", required=True, help="Path to CSV data file")
    parser.add_argument(
        "--n_trials", type=int, default=100, help="Number of optimization trials"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.3, help="Test set size for validation"
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random state for reproducibility"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}")
    data = pd.read_csv(args.data_file)
    X = data.values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_val = train_test_split(
        X_scaled, test_size=args.test_size, random_state=args.random_state
    )

    print(f"Training set size: {X_train.shape}")
    print(f"Validation set size: {X_val.shape}")

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Create objective function
    if args.algorithm == "isolation_forest":
        objective = create_isolation_forest_objective(X_train, X_val)
    else:  # one_class_svm
        objective = create_one_class_svm_objective(X_train, X_val)

    # Optimize
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials)

    # Print results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)

    print(f"Best value: {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )

    # Show top 5 trials
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nlargest(5, "value")
    print(
        top_trials[
            (
                ["number", "value", "params_n_estimators", "params_contamination"]
                if args.algorithm == "isolation_forest"
                else ["number", "value", "params_kernel", "params_nu"]
            )
        ]
    )


if __name__ == "__main__":
    main()
