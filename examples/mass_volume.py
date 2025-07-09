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
y_test = np.hstack([
    np.zeros(len(x_test_normal)),
    np.ones(len(x_anomaly))
])


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_samples': trial.suggest_float('max_samples', 0.1, 1.0),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
    }

    cv_scores = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(x_train_normal):
        model = IsolationForest(**params, contamination=sys.float_info.min, random_state=42)
        model.fit(x_train_normal[train_idx])
        
        val_scores = -model.score_samples(x_train_normal[val_idx])
        mv_result = labelfree.mass_volume_auc(val_scores, x_train_normal[val_idx], n_thresholds=50, random_state=42)
        cv_scores.append(mv_result["auc"])
    
    model = IsolationForest(**params, contamination=sys.float_info.min, random_state=42)
    model.fit(x_train_normal)
    test_scores = -model.score_samples(x_test)
    
    roc_auc = roc_auc_score(y_test, test_scores)
    pr_auc = average_precision_score(y_test, test_scores)
    mv_auc = labelfree.mass_volume_auc(test_scores, x_test, n_thresholds=50, random_state=42)["auc"]
    
    trial.set_user_attr('roc_auc', roc_auc)
    trial.set_user_attr('pr_auc', pr_auc)
    trial.set_user_attr('mv_auc', mv_auc)
    
    return np.mean(cv_scores)


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
roc_aucs = np.array([t.user_attrs['roc_auc'] for t in completed_trials])
pr_aucs = np.array([t.user_attrs['pr_auc'] for t in completed_trials])
mv_aucs = np.array([t.user_attrs['mv_auc'] for t in completed_trials])

print("="*60)
print("PROXY QUALITY ANALYSIS: Mass-Volume vs Supervised Metrics")
print("="*60)

# 1. Correlation Analysis
print("\n1. CORRELATION ANALYSIS")
print("-" * 25)
pearson_roc_mv, p_pearson_roc = pearsonr(roc_aucs, mv_aucs)
pearson_pr_mv, p_pearson_pr = pearsonr(pr_aucs, mv_aucs)
spearman_roc_mv, p_spearman_roc = spearmanr(roc_aucs, mv_aucs)
spearman_pr_mv, p_spearman_pr = spearmanr(pr_aucs, mv_aucs)
kendall_roc_mv, p_kendall_roc = kendalltau(roc_aucs, mv_aucs)
kendall_pr_mv, p_kendall_pr = kendalltau(pr_aucs, mv_aucs)

print(f"ROC-AUC vs MV-AUC:")
print(f"  Pearson:  {pearson_roc_mv:.3f} (p={p_pearson_roc:.3f})")
print(f"  Spearman: {spearman_roc_mv:.3f} (p={p_spearman_roc:.3f})")
print(f"  Kendall:  {kendall_roc_mv:.3f} (p={p_kendall_roc:.3f})")

print(f"\nPR-AUC vs MV-AUC:")
print(f"  Pearson:  {pearson_pr_mv:.3f} (p={p_pearson_pr:.3f})")
print(f"  Spearman: {spearman_pr_mv:.3f} (p={p_spearman_pr:.3f})")
print(f"  Kendall:  {kendall_pr_mv:.3f} (p={p_kendall_pr:.3f})")

# 2. Ranking Agreement
print("\n2. RANKING AGREEMENT")
print("-" * 20)
roc_ranks = rankdata(-roc_aucs)  # Higher ROC-AUC is better
pr_ranks = rankdata(-pr_aucs)    # Higher PR-AUC is better
mv_ranks = rankdata(mv_aucs)     # Lower MV-AUC is better

rank_corr_roc_mv = spearmanr(roc_ranks, mv_ranks)[0]
rank_corr_pr_mv = spearmanr(pr_ranks, mv_ranks)[0]

print(f"Rank correlation ROC-AUC vs MV-AUC: {rank_corr_roc_mv:.3f}")
print(f"Rank correlation PR-AUC vs MV-AUC:  {rank_corr_pr_mv:.3f}")

# 3. Top-K Overlap Analysis
print("\n3. TOP-K OVERLAP ANALYSIS")
print("-" * 26)
for k in [5, 10, 20]:
    top_k_roc = set(np.argsort(roc_aucs)[-k:])
    top_k_pr = set(np.argsort(pr_aucs)[-k:])
    top_k_mv = set(np.argsort(mv_aucs)[:k])
    
    overlap_roc_mv = len(top_k_roc & top_k_mv) / k
    overlap_pr_mv = len(top_k_pr & top_k_mv) / k
    
    print(f"Top-{k} overlap ROC-AUC vs MV-AUC: {overlap_roc_mv:.3f}")
    print(f"Top-{k} overlap PR-AUC vs MV-AUC:  {overlap_pr_mv:.3f}")

# 4. Proxy Quality Metrics
print("\n4. PROXY QUALITY METRICS")
print("-" * 25)
best_roc_idx = np.argmax(roc_aucs)
best_pr_idx = np.argmax(pr_aucs)
best_mv_idx = np.argmin(mv_aucs)

roc_regret = roc_aucs[best_roc_idx] - roc_aucs[best_mv_idx]
pr_regret = pr_aucs[best_pr_idx] - pr_aucs[best_mv_idx]

print(f"Best ROC-AUC: {roc_aucs[best_roc_idx]:.3f}")
print(f"MV-selected ROC-AUC: {roc_aucs[best_mv_idx]:.3f}")
print(f"ROC-AUC regret: {roc_regret:.3f}")

print(f"\nBest PR-AUC: {pr_aucs[best_pr_idx]:.3f}")
print(f"MV-selected PR-AUC: {pr_aucs[best_mv_idx]:.3f}")
print(f"PR-AUC regret: {pr_regret:.3f}")

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

roc_assessment = assess_proxy_quality(abs(spearman_roc_mv), len(set(np.argsort(roc_aucs)[-10:]) & set(np.argsort(mv_aucs)[:10]))/10, roc_regret)
pr_assessment = assess_proxy_quality(abs(spearman_pr_mv), len(set(np.argsort(pr_aucs)[-10:]) & set(np.argsort(mv_aucs)[:10]))/10, pr_regret)

print(f"MV-AUC as proxy for ROC-AUC: {roc_assessment}")
print(f"MV-AUC as proxy for PR-AUC:  {pr_assessment}")

print(f"\nBest parameters (MV-AUC selected):")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")