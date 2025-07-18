import sys
import optuna
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr, kendalltau, rankdata
import labelfree

x, y = load_breast_cancer(return_X_y=True)

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
        sireos_score = labelfree.sireos_separation(
            val_scores, x_train_normal[val_idx], random_state=42
        )
        cv_scores.append(sireos_score)

    model = IsolationForest(**params, contamination=sys.float_info.min, random_state=42)
    model.fit(x_train_normal)
    test_scores = -model.score_samples(x_test)

    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    sireos_auc = labelfree.sireos_separation(test_scores, x_test, random_state=42)
    sireos_sep = labelfree.sireos_separation(test_scores, x_test, random_state=42)

    trial.set_user_attr("roc_auc", roc_auc)
    trial.set_user_attr("pr_auc", pr_auc)
    trial.set_user_attr("sireos_auc", sireos_auc)
    trial.set_user_attr("sireos_sep", sireos_sep)

    return np.mean(cv_scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

completed_trials = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
]
roc_aucs = np.array([t.user_attrs["roc_auc"] for t in completed_trials])
pr_aucs = np.array([t.user_attrs["pr_auc"] for t in completed_trials])
sireos_aucs = np.array([t.user_attrs["sireos_auc"] for t in completed_trials])
sireos_seps = np.array([t.user_attrs["sireos_sep"] for t in completed_trials])

print("=" * 60)
print("PROXY QUALITY ANALYSIS: SIREOS vs Supervised Metrics")
print("=" * 60)

# 1. Correlation Analysis
print("\n1. CORRELATION ANALYSIS")
print("-" * 25)
pearson_roc_sireos, p_pearson_roc = pearsonr(roc_aucs, sireos_aucs)
pearson_pr_sireos, p_pearson_pr = pearsonr(pr_aucs, sireos_aucs)
spearman_roc_sireos, p_spearman_roc = spearmanr(roc_aucs, sireos_aucs)
spearman_pr_sireos, p_spearman_pr = spearmanr(pr_aucs, sireos_aucs)
kendall_roc_sireos, p_kendall_roc = kendalltau(roc_aucs, sireos_aucs)
kendall_pr_sireos, p_kendall_pr = kendalltau(pr_aucs, sireos_aucs)

print("ROC-AUC vs SIREOS:")
print(f"  Pearson:  {pearson_roc_sireos:.3f} (p={p_pearson_roc:.3f})")
print(f"  Spearman: {spearman_roc_sireos:.3f} (p={p_spearman_roc:.3f})")
print(f"  Kendall:  {kendall_roc_sireos:.3f} (p={p_kendall_roc:.3f})")

print("\nPR-AUC vs SIREOS:")
print(f"  Pearson:  {pearson_pr_sireos:.3f} (p={p_pearson_pr:.3f})")
print(f"  Spearman: {spearman_pr_sireos:.3f} (p={p_spearman_pr:.3f})")
print(f"  Kendall:  {kendall_pr_sireos:.3f} (p={p_kendall_pr:.3f})")

# SIREOS Separation correlations
print("\n--- SIREOS SEPARATION CORRELATIONS ---")
pearson_roc_sep, p_pearson_roc_sep = pearsonr(roc_aucs, sireos_seps)
pearson_pr_sep, p_pearson_pr_sep = pearsonr(pr_aucs, sireos_seps)
spearman_roc_sep, p_spearman_roc_sep = spearmanr(roc_aucs, sireos_seps)
spearman_pr_sep, p_spearman_pr_sep = spearmanr(pr_aucs, sireos_seps)

print("ROC-AUC vs SIREOS Separation:")
print(f"  Pearson:  {pearson_roc_sep:.3f} (p={p_pearson_roc_sep:.3f})")
print(f"  Spearman: {spearman_roc_sep:.3f} (p={p_spearman_roc_sep:.3f})")

print("\nPR-AUC vs SIREOS Separation:")
print(f"  Pearson:  {pearson_pr_sep:.3f} (p={p_pearson_pr_sep:.3f})")
print(f"  Spearman: {spearman_pr_sep:.3f} (p={p_spearman_pr_sep:.3f})")

# 2. Ranking Agreement
print("\n2. RANKING AGREEMENT")
print("-" * 20)
roc_ranks = rankdata(-roc_aucs)  # Higher ROC-AUC is better
pr_ranks = rankdata(-pr_aucs)  # Higher PR-AUC is better
sireos_ranks = rankdata(-sireos_aucs)  # Higher SIREOS is better
sireos_sep_ranks = rankdata(-sireos_seps)  # Higher SIREOS separation is better

rank_corr_roc_sireos = spearmanr(roc_ranks, sireos_ranks)[0]
rank_corr_pr_sireos = spearmanr(pr_ranks, sireos_ranks)[0]
rank_corr_roc_sireos_sep = spearmanr(roc_ranks, sireos_sep_ranks)[0]
rank_corr_pr_sireos_sep = spearmanr(pr_ranks, sireos_sep_ranks)[0]

print(f"Rank correlation ROC-AUC vs SIREOS:           {rank_corr_roc_sireos:.3f}")
print(f"Rank correlation PR-AUC vs SIREOS:            {rank_corr_pr_sireos:.3f}")
print(f"Rank correlation ROC-AUC vs SIREOS Separation: {rank_corr_roc_sireos_sep:.3f}")
print(f"Rank correlation PR-AUC vs SIREOS Separation:  {rank_corr_pr_sireos_sep:.3f}")

# 3. Top-K Overlap Analysis
print("\n3. TOP-K OVERLAP ANALYSIS")
print("-" * 26)
for k in [5, 10, 20]:
    top_k_roc = set(np.argsort(roc_aucs)[-k:])
    top_k_pr = set(np.argsort(pr_aucs)[-k:])
    top_k_sireos = set(np.argsort(sireos_aucs)[-k:])
    top_k_sireos_sep = set(np.argsort(sireos_seps)[-k:])

    overlap_roc_sireos = len(top_k_roc & top_k_sireos) / k
    overlap_pr_sireos = len(top_k_pr & top_k_sireos) / k
    overlap_roc_sireos_sep = len(top_k_roc & top_k_sireos_sep) / k
    overlap_pr_sireos_sep = len(top_k_pr & top_k_sireos_sep) / k

    print(f"Top-{k} overlap ROC-AUC vs SIREOS:           {overlap_roc_sireos:.3f}")
    print(f"Top-{k} overlap PR-AUC vs SIREOS:            {overlap_pr_sireos:.3f}")
    print(f"Top-{k} overlap ROC-AUC vs SIREOS Separation: {overlap_roc_sireos_sep:.3f}")
    print(f"Top-{k} overlap PR-AUC vs SIREOS Separation:  {overlap_pr_sireos_sep:.3f}")

# 4. Proxy Quality Metrics
print("\n4. PROXY QUALITY METRICS")
print("-" * 25)
best_roc_idx = np.argmax(roc_aucs)
best_pr_idx = np.argmax(pr_aucs)
best_sireos_idx = np.argmax(sireos_aucs)
best_sireos_sep_idx = np.argmax(sireos_seps)

roc_regret_sireos = roc_aucs[best_roc_idx] - roc_aucs[best_sireos_idx]
pr_regret_sireos = pr_aucs[best_pr_idx] - pr_aucs[best_sireos_idx]
roc_regret_sireos_sep = roc_aucs[best_roc_idx] - roc_aucs[best_sireos_sep_idx]
pr_regret_sireos_sep = pr_aucs[best_pr_idx] - pr_aucs[best_sireos_sep_idx]

print(f"Best ROC-AUC: {roc_aucs[best_roc_idx]:.3f}")
print(f"SIREOS-selected ROC-AUC: {roc_aucs[best_sireos_idx]:.3f}")
print(f"SIREOS ROC-AUC regret: {roc_regret_sireos:.3f}")
print(f"SIREOS Separation-selected ROC-AUC: {roc_aucs[best_sireos_sep_idx]:.3f}")
print(f"SIREOS Separation ROC-AUC regret: {roc_regret_sireos_sep:.3f}")

print(f"\nBest PR-AUC: {pr_aucs[best_pr_idx]:.3f}")
print(f"SIREOS-selected PR-AUC: {pr_aucs[best_sireos_idx]:.3f}")
print(f"SIREOS PR-AUC regret: {pr_regret_sireos:.3f}")
print(f"SIREOS Separation-selected PR-AUC: {pr_aucs[best_sireos_sep_idx]:.3f}")
print(f"SIREOS Separation PR-AUC regret: {pr_regret_sireos_sep:.3f}")

# 5. Summary Assessment
print("\n5. PROXY QUALITY SUMMARY")
print("-" * 24)


def assess_proxy_quality(corr, overlap_10, regret):
    if corr > 0.7 and overlap_10 > 0.7 and regret < 0.05:
        return "EXCELLENT"
    elif corr > 0.5 and overlap_10 > 0.5 and regret < 0.1:
        return "GOOD"
    elif corr > 0.3 and overlap_10 > 0.3 and regret < 0.15:
        return "MODERATE"
    else:
        return "POOR"


roc_assessment_sireos = assess_proxy_quality(
    abs(spearman_roc_sireos),
    len(set(np.argsort(roc_aucs)[-10:]) & set(np.argsort(sireos_aucs)[-10:])) / 10,
    roc_regret_sireos,
)
pr_assessment_sireos = assess_proxy_quality(
    abs(spearman_pr_sireos),
    len(set(np.argsort(pr_aucs)[-10:]) & set(np.argsort(sireos_aucs)[-10:])) / 10,
    pr_regret_sireos,
)

roc_assessment_sireos_sep = assess_proxy_quality(
    abs(spearman_roc_sep),
    len(set(np.argsort(roc_aucs)[-10:]) & set(np.argsort(sireos_seps)[-10:])) / 10,
    roc_regret_sireos_sep,
)
pr_assessment_sireos_sep = assess_proxy_quality(
    abs(spearman_pr_sep),
    len(set(np.argsort(pr_aucs)[-10:]) & set(np.argsort(sireos_seps)[-10:])) / 10,
    pr_regret_sireos_sep,
)

print(f"SIREOS as proxy for ROC-AUC:             {roc_assessment_sireos}")
print(f"SIREOS as proxy for PR-AUC:              {pr_assessment_sireos}")
print(f"SIREOS Separation as proxy for ROC-AUC:  {roc_assessment_sireos_sep}")
print(f"SIREOS Separation as proxy for PR-AUC:   {pr_assessment_sireos_sep}")

# 6. Method Comparison
print("\n6. METHOD COMPARISON")
print("-" * 19)
print("SIREOS vs SIREOS Separation:")
sireos_sireos_sep_corr = spearmanr(sireos_aucs, sireos_seps)[0]
print(f"  Spearman correlation: {sireos_sireos_sep_corr:.3f}")

print(f"\nMean SIREOS score: {sireos_aucs.mean():.3f} ± {sireos_aucs.std():.3f}")
print(f"Mean SIREOS Separation score: {sireos_seps.mean():.3f} ± {sireos_seps.std():.3f}")

print("\nBest parameters (SIREOS-selected):")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")