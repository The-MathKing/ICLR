import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('phase7_results/truthfulqa_smollm_features.csv')
df['target'] = (df['label'] == 'hallucinated').astype(int)
y = df['target']

X_0d = df[[c for c in df.columns if 'h0' in c]]
X_1d = df[[c for c in df.columns if 'h1' in c]]
X_comb = df[[c for c in df.columns if 'h0' in c or 'h1' in c]]
msp = df['msp_score'].values

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

auc_msp = []
auc_0d_lr = []
auc_1d_lr = []
auc_comb_lr = []

for train_idx, test_idx in skf.split(X_0d, y):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    lr = LogisticRegression()
    lr.fit(msp[train_idx].reshape(-1, 1), y_train)
    preds = lr.predict_proba(msp[test_idx].reshape(-1, 1))[:, 1]
    auc_msp.append(roc_auc_score(y_test, preds))
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_0d.iloc[train_idx], y_train)
    auc_0d_lr.append(roc_auc_score(y_test, lr.predict_proba(X_0d.iloc[test_idx])[:, 1]))
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_1d.iloc[train_idx], y_train)
    auc_1d_lr.append(roc_auc_score(y_test, lr.predict_proba(X_1d.iloc[test_idx])[:, 1]))
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_comb.iloc[train_idx], y_train)
    auc_comb_lr.append(roc_auc_score(y_test, lr.predict_proba(X_comb.iloc[test_idx])[:, 1]))

print(f"MSP Baseline (LR): {np.mean(auc_msp):.3f} +/- {np.std(auc_msp):.3f}")
print(f"0D Only (LR):      {np.mean(auc_0d_lr):.3f} +/- {np.std(auc_0d_lr):.3f}")
print(f"1D Only (LR):      {np.mean(auc_1d_lr):.3f} +/- {np.std(auc_1d_lr):.3f}")
print(f"Combined (LR):     {np.mean(auc_comb_lr):.3f} +/- {np.std(auc_comb_lr):.3f}")

from scipy import stats
_, p_val = stats.ttest_rel(auc_0d_lr, auc_msp)
print(f"Paired t-test (0D LR vs MSP LR): p-value = {p_val:.4f}")
