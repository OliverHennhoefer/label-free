"""
IREOS Evaluation Example with Proper Score Normalization

This example demonstrates proper usage of IREOS for hyperparameter tuning
in anomaly detection. Key corrections implemented:

1. **Score Normalization**: Raw IsolationForest scores are normalized using
   labelfree.auto_normalize_scores() to convert them to [0,1] probability-like
   values as required by the original IREOS specification.

2. **Algorithm-Specific Handling**: IsolationForest returns negative scores
   where lower values indicate higher anomaly probability, so scores are
   automatically inverted during normalization.

3. **Proper IREOS Usage**: While this example uses IREOS for hyperparameter
   tuning, note that IREOS was originally designed for comparing different
   outlier detection algorithms rather than tuning a single algorithm.

Without score normalization, IREOS returns epsilon values (~2e-16) because
the separability computation becomes meaningless with raw scores.
"""

import sys
import optuna
import numpy as np
import warnings

from sklearn.datasets import fetch_openml
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import labelfree

# Filter CVXPY warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="cvxpy")

rng = np.random.default_rng(seed=42)

x, y = fetch_openml("shuttle", version=1, return_X_y=True, as_frame=False)

# Convert to numeric and handle any missing values
x = x.astype(np.float32)
y = y.astype(int)

# Normalize the data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

# Class 1 = normal operations, classes 2-7 = various anomalies
x_normal = x[y == 1]
x_anomaly = x[y != 1]

x_train_normal, x_test_normal = train_test_split(
    x_normal, test_size=0.3, random_state=42
)

x_test = np.vstack([x_test_normal, x_anomaly])
y_test = np.hstack([np.zeros(len(x_test_normal)), np.ones(len(x_anomaly))])


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_samples": trial.suggest_float("max_samples", 0.1, 1.0),
        "max_features": trial.suggest_float("max_features", 0.1, 1.0),
        "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
    }

    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(x_train_normal):
        model = IsolationForest(
            **params, contamination=sys.float_info.min, random_state=42
        )
        model.fit(x_train_normal[train_idx])

        # Use validation data to compute a proxy metric for IREOS
        # Since we can't use IREOS directly on normal-only data, we'll use
        # score variance as a proxy (higher variance suggests better separability)
        val_scores = -model.score_samples(x_train_normal[val_idx])
        
        # Score variance as proxy for separability potential
        # Models with higher score variance on normal data may better distinguish anomalies
        score_variance = np.var(val_scores)
        cv_scores.append(score_variance)

    # Final evaluation on mixed test data with proper IREOS
    model = IsolationForest(**params, contamination=sys.float_info.min, random_state=42)
    model.fit(x_train_normal)
    test_scores = -model.score_samples(x_test)

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    
    # Normalize test scores for proper IREOS evaluation
    test_scores_normalized = labelfree.auto_normalize_scores(test_scores, "isolation_forest")
    # Use IREOS on mixed test data (this will produce meaningful results)
    ireos_auc, _ = labelfree.ireos(
        test_scores_normalized,
        x_test,
        classifier="klr",  # Use KLR classifier
        n_gamma=20,
        n_monte_carlo=3000,
        random_state=42,
    )

    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("pr_auc", pr_auc)
    trial.set_user_attr("ireos_auc", ireos_auc)

    # Return score variance proxy for hyperparameter optimization
    return np.mean(cv_scores)


# Create reproducible study with fixed sampler
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)
study.optimize(objective, n_trials=100)

completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
roc_aucs = np.array([t.user_attrs["roc_auc"] for t in completed_trials])
pr_aucs = np.array([t.user_attrs["pr_auc"] for t in completed_trials])
ireos_aucs = np.array([t.user_attrs["ireos_auc"] for t in completed_trials])

# Results
spearman_roc_ireos = spearmanr(roc_aucs, ireos_aucs)[0]
spearman_pr_ireos = spearmanr(pr_aucs, ireos_aucs)[0]

print("\n" + "=" * 50)
print("IREOS Evaluation with Score Normalization")
print("=" * 50)
print(f"IREOS vs ROC-AUC correlation: {spearman_roc_ireos:.3f}")
print(f"IREOS vs PR-AUC correlation:  {spearman_pr_ireos:.3f}")
print("\nBest parameters (selected by score variance proxy):")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
print("\nNote: CV used score variance as IREOS proxy (normal-only data)")
print("      Final IREOS evaluation performed on mixed test data")

# Demonstrate different classifiers on best model
print("\n" + "=" * 50)
print("Classifier Comparison on Best Model")
print("=" * 50)

best_params = study.best_trial.params
best_model = IsolationForest(
    **best_params, contamination=sys.float_info.min, random_state=42
)
best_model.fit(x_train_normal)
best_test_scores = -best_model.score_samples(x_test)
# Normalize scores for classifier comparison
best_test_scores_normalized = labelfree.auto_normalize_scores(best_test_scores, "isolation_forest")

classifiers = ["logistic", "klr", "svm", "knn"]
for classifier in classifiers:
    try:
        ireos_score, p_value = labelfree.ireos(
            best_test_scores_normalized,
            x_test,
            classifier=classifier,
            n_gamma=15,
            n_monte_carlo=20,
            random_state=42,
        )
        print(f"{classifier:8}: IREOS={ireos_score:.4f}, p-value={p_value:.4f}")
    except Exception as e:
        print(f"{classifier:8}: Error - {str(e)[:50]}...")
