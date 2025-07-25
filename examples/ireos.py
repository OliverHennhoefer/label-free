import sys
import optuna
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import labelfree

rng = np.random.default_rng(seed=42)

x, y = load_breast_cancer(return_X_y=True)

# Normalize the data
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x_normal = x[y == 0]
x_anomaly = x[y == 1]

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

        val_scores = -model.score_samples(x_train_normal[val_idx])
        # Use faster parameters for hyperparameter optimization
        ireos_score, _ = labelfree.ireos(
            val_scores,
            x_train_normal[val_idx],
            n_gamma=15,
            n_monte_carlo=20,
            random_state=42,
        )
        cv_scores.append(ireos_score)

    model = IsolationForest(**params, contamination=sys.float_info.min, random_state=42)
    model.fit(x_train_normal)
    test_scores = -model.score_samples(x_test)

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    # Use faster parameters for final evaluation
    ireos_auc, _ = labelfree.ireos(
        test_scores, x_test, n_gamma=20, n_monte_carlo=30, random_state=42
    )

    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("pr_auc", pr_auc)
    trial.set_user_attr("ireos_auc", ireos_auc)

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
print("IREOS Evaluation")
print("=" * 50)
print(f"IREOS vs ROC-AUC correlation: {spearman_roc_ireos:.3f}")
print(f"IREOS vs PR-AUC correlation:  {spearman_pr_ireos:.3f}")
print("\nBest parameters (IREOS selected):")
for key, value in study.best_trial.params.items():
    print(f"  {key}: {value}")
