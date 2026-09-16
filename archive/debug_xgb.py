import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

df = pd.read_csv("phase7_results/truthfulqa_smollm_features.csv")
df['target'] = (df['label'] == 'hallucinated').astype(int)
y = df['target']
X_1d = df[[c for c in df.columns if 'h1' in c]]
groups = df['example_id']

skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(skf.split(X_1d, y, groups=groups)):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb.fit(X_1d.iloc[train_idx], y_train)
    
    preds = xgb.predict_proba(X_1d.iloc[test_idx])[:, 1]
    auc = roc_auc_score(y_test, preds)
    print(f"Fold {fold}: AUC = {auc:.3f}")
