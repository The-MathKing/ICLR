import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def get_length_matched_subset(df, bin_size=10):
    """
    Given a dataframe with 'label' and 'n_answer_tokens', 
    return a subset where the distribution of n_answer_tokens 
    is perfectly matched between the two classes by exact binning.
    """
    df = df.copy()
    df['length_bin'] = (df['n_answer_tokens'] // bin_size) * bin_size
    
    matched_indices = []
    
    # We want pairs of grounded and hallucinated from the SAME length bin
    for bin_val in df['length_bin'].unique():
        bin_df = df[df['length_bin'] == bin_val]
        grounded_idx = bin_df[bin_df['label'] == 'grounded'].index.tolist()
        hallu_idx = bin_df[bin_df['label'] == 'hallucinated'].index.tolist()
        
        min_count = min(len(grounded_idx), len(hallu_idx))
        if min_count > 0:
            # randomly sample to match counts
            np.random.shuffle(grounded_idx)
            np.random.shuffle(hallu_idx)
            matched_indices.extend(grounded_idx[:min_count])
            matched_indices.extend(hallu_idx[:min_count])
            
    return df.loc[matched_indices]

def run_length_matched_eval(csv_path="phase1_controls_halueval.csv"):
    df = pd.read_csv(csv_path)
    matched_df = get_length_matched_subset(df)
    
    print(f"Original size: {len(df)}, Matched size: {len(matched_df)}")
    
    y = (matched_df['label'] == 'hallucinated').astype(int)
    groups = matched_df['example_id']
    
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    def eval_features(X, name):
        aucs = []
        for train_idx, test_idx in skf.split(X, y, groups=groups):
            lr = LogisticRegression(max_iter=1000)
            lr.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = lr.predict_proba(X.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y.iloc[test_idx], preds))
        mean_auc = np.mean(aucs)
        print(f"{name:35s}: {mean_auc:.3f}")
        return mean_auc
        
    print("\n=== Phase 1.2 Length-Matched Evaluations ===")
    
    # Length (should be ~0.50 if perfectly matched)
    X_len = matched_df[['n_prompt_tokens', 'n_answer_tokens', 'log_n_answer_tokens']]
    eval_features(X_len, "Length-Only Baseline (Matched)")
    
    # MST Total Weight (0D proxy)
    X_mst = matched_df[[c for c in matched_df.columns if 'mst_weight' in c]]
    eval_features(X_mst, "MST Total Weight (0D proxy)")
    
    # Non-Topological Stats
    X_non_top = matched_df[[c for c in matched_df.columns if any(x in c for x in ['mean_attn', 'max_attn', 'sink_mass'])]]
    eval_features(X_non_top, "Non-Topological Stats")
    
    # Hidden-State Linear Probe
    X_hidden = matched_df[[c for c in matched_df.columns if 'hidden_' in c]]
    eval_features(X_hidden, "Hidden-State Probe")

if __name__ == "__main__":
    run_length_matched_eval()
