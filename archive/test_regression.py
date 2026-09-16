import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def test_regression():
    # Create synthetic dataset
    N = 1000
    example_ids = np.repeat(np.arange(N//2), 2)
    labels = np.tile([0, 1], N//2)
    
    df = pd.DataFrame({'example_id': example_ids, 'label': labels})
    
    # Synthetic perfect feature: completely correlates with label, slightly noisy
    df['synthetic_feature'] = df['label'] + np.random.normal(0, 0.1, N)
    
    X = df[['synthetic_feature']]
    y = df['label']
    groups = df['example_id']
    
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    aucs = []
    for train_idx, test_idx in skf.split(X, y, groups=groups):
        lr = LogisticRegression()
        lr.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = lr.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], preds))
        
    mean_auc = np.mean(aucs)
    print(f"End-to-End Regression Test AUC: {mean_auc:.4f}")
    assert mean_auc > 0.99, "AUC should be near 1.0 for a perfect feature!"
    print("Regression test PASSED.")

if __name__ == '__main__':
    test_regression()
