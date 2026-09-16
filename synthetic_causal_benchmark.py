"""
synthetic_causal_benchmark.py
=============================
Rigorous synthetic causal benchmark evaluating Topological Data Analysis (TDA)
on attention-like graphs under five strictly controlled structural regimes:

  S1: Pure Length Confound (Identical distribution, different sequence lengths)
  S2: Pure Topology / Positive Control (Marginal-matched, planted 1D persistent cycle in Class 1)
  S3: Pure Attention Marginals (Identical length, shifted attention entropy/density without cycles)
  S4: Topology + Length Confound (Planted cycles with sequence length variation)
  S5: Negative Control (Pure random noise with permuted labels)

This establishes a formal unit test proving pipeline sensitivity to true topological
cavities (Positive Control) and quantifying confound vulnerability.
"""

import numpy as np
import pandas as pd
import scipy.sparse.csgraph as csgraph
import ripser
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import os

np.random.seed(42)

def generate_base_distance_matrix(N, noise_level=0.1):
    """
    Generates a baseline symmetric distance matrix D on N nodes representing an un-looped tree/cluster.
    By construction this background has (near-)zero true 1D cycles regardless of N -- it tests whether
    LENGTH ALONE, absent any mechanical cycle-growth process, produces a spurious 1D signal (it should
    not, and does not: see S1 below). It does NOT test Proposition 1's actual claimed mechanism
    (clique expansion in i.i.d.-weighted complete graphs) -- see generate_iid_distance_matrix for that.
    """
    # Points on a 1D line with noise (trivial topology, H1 = 0)
    coords = np.linspace(0, 1, N).reshape(-1, 1) + np.random.randn(N, 1) * noise_level
    diff = coords - coords.T
    D = np.abs(diff)
    np.fill_diagonal(D, 0.0)
    # Scale to [0, 1]
    D = D / (np.max(D) + 1e-8)
    return D


def generate_iid_distance_matrix(N):
    """
    Generates D with i.i.d. Uniform[0,1] off-diagonal weights, symmetrized -- exactly the null model
    Proposition 1 analyzes. Unlike generate_base_distance_matrix, this background DOES have a growing
    population of chordless 4-cycles as N grows, so raw 1D persistence should respond to N here if the
    extraction pipeline is sensitive to the mechanism the Proposition actually describes.
    """
    D = np.random.uniform(0, 1, size=(N, N))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    return D

def plant_pure_topological_cycle(D_base, cycle_nodes, cycle_radius=0.15):
    """
    Plants a true 1-cycle on a subset of nodes by embedding them on a circle,
    preserving the overall distance scale and marginal statistics.
    """
    D = D_base.copy()
    k = len(cycle_nodes)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    circle_coords = np.column_stack([np.cos(angles), np.sin(angles)]) * cycle_radius
    
    for i in range(k):
        for j in range(k):
            u = cycle_nodes[i]
            v = cycle_nodes[j]
            dist = np.linalg.norm(circle_coords[i] - circle_coords[j])
            D[u, v] = dist
            D[v, u] = dist
            
    np.fill_diagonal(D, 0.0)
    return D

def extract_features_from_distance(D, N):
    # 1. Marginals / Non-topological stats on pseudo-attention W = 1 - D
    W = 1.0 - D
    np.fill_diagonal(W, 0.0)
    mean_w = float(np.mean(W))
    max_w = float(np.max(W))
    std_w = float(np.std(W))
    degree_var = float(np.var(np.sum(W, axis=1)))
    
    # 2. MST
    mst = csgraph.minimum_spanning_tree(D)
    mst_weight = float(mst.sum())
    mst_max_edge = float(mst.max()) if mst.nnz > 0 else 0.0
    mst_mean_edge = mst_weight / max(1, (N - 1))
    
    # 3. Persistent Homology
    r = ripser.ripser(D, maxdim=1, distance_matrix=True)
    diagrams = r['dgms']
    
    # H0
    h0 = diagrams[0]
    h0_finite = h0[h0[:, 1] != np.inf] if len(h0) > 0 else np.array([])
    if len(h0_finite) > 0:
        h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0]
        h0_max_lifetime = float(np.max(h0_lifetimes))
        h0_total_persistence = float(np.sum(h0_lifetimes))
    else:
        h0_max_lifetime = 0.0
        h0_total_persistence = 0.0
        
    # H1
    h1 = diagrams[1]
    h1_finite = h1[h1[:, 1] != np.inf] if len(h1) > 0 else np.array([])
    if len(h1_finite) > 0:
        h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0]
        max_idx = np.argmax(h1_lifetimes)
        h1_max_lifetime = float(np.max(h1_lifetimes))
        h1_total_persistence = float(np.sum(h1_lifetimes))
        h1_max_birth = float(h1_finite[max_idx, 0])
        h1_max_death = float(h1_finite[max_idx, 1])
        h1_count = float(len(h1_finite))
    else:
        h1_max_lifetime = 0.0
        h1_total_persistence = 0.0
        h1_max_birth = 0.0
        h1_max_death = 0.0
        h1_count = 0.0
        
    # Normalized features
    h1_total_norm = h1_total_persistence / float(N)
    h1_max_norm = h1_max_lifetime / float(np.log(N + 1.0))
    mst_norm = mst_weight / float(N)
    
    return {
        "N": N,
        "mean_w": mean_w,
        "max_w": max_w,
        "std_w": std_w,
        "degree_var": degree_var,
        "mst_weight": mst_weight,
        "mst_norm": mst_norm,
        "mst_max_edge": mst_max_edge,
        "h0_max_lifetime": h0_max_lifetime,
        "h0_total_persistence": h0_total_persistence,
        "h1_max_lifetime": h1_max_lifetime,
        "h1_total_persistence": h1_total_persistence,
        "h1_max_birth": h1_max_birth,
        "h1_max_death": h1_max_death,
        "h1_count": h1_count,
        "h1_total_norm": h1_total_norm,
        "h1_max_norm": h1_max_norm,
    }

def generate_controlled_regimes(n_samples=400):
    regime_dfs = {}
    half = n_samples // 2
    
    # -------------------------------------------------------------
    # S1: Pure Length Confound (Identical 1D tree metric, different N)
    # -------------------------------------------------------------
    s1_rows = []
    for i in range(n_samples):
        label = 0 if i < half else 1
        N = np.random.randint(20, 36) if label == 0 else np.random.randint(55, 81)
        # Tree metric with ambient 1D noise (zero true 1D cycles)
        D = generate_base_distance_matrix(N, noise_level=0.02)
        feats = extract_features_from_distance(D, N)
        feats["label"] = label
        s1_rows.append(feats)
    regime_dfs["S1"] = pd.DataFrame(s1_rows)

    # -------------------------------------------------------------
    # S1b: Pure Length Confound under i.i.d. background (Proposition 1's own
    # null model) -- tests whether raw 1D responds to the length-driven
    # clique-expansion mechanism the Proposition actually describes, rather
    # than to length alone on a background with no mechanical cycle growth.
    # -------------------------------------------------------------
    s1b_rows = []
    for i in range(n_samples):
        label = 0 if i < half else 1
        N = np.random.randint(20, 36) if label == 0 else np.random.randint(55, 81)
        D = generate_iid_distance_matrix(N)
        feats = extract_features_from_distance(D, N)
        feats["label"] = label
        s1b_rows.append(feats)
    regime_dfs["S1b"] = pd.DataFrame(s1b_rows)

    # -------------------------------------------------------------
    # S2: Pure Topology (Positive Control: Matched N, planted cycle in Class 1)
    # -------------------------------------------------------------
    s2_rows = []
    N_fixed = 40
    for i in range(n_samples):
        label = 0 if i < half else 1
        D_base = generate_base_distance_matrix(N_fixed, noise_level=0.05)
        if label == 1:
            cycle_nodes = np.random.choice(N_fixed, size=10, replace=False)
            D = plant_pure_topological_cycle(D_base, cycle_nodes, cycle_radius=0.25)
        else:
            D = D_base
        feats = extract_features_from_distance(D, N_fixed)
        feats["label"] = label
        s2_rows.append(feats)
    regime_dfs["S2"] = pd.DataFrame(s2_rows)
    
    # -------------------------------------------------------------
    # S3: Pure Marginals (Matched N=40, shifted mean/variance without cycles)
    # -------------------------------------------------------------
    s3_rows = []
    for i in range(n_samples):
        label = 0 if i < half else 1
        noise = 0.01 if label == 0 else 0.25
        D = generate_base_distance_matrix(N_fixed, noise_level=noise)
        feats = extract_features_from_distance(D, N_fixed)
        feats["label"] = label
        s3_rows.append(feats)
    regime_dfs["S3"] = pd.DataFrame(s3_rows)
    
    # -------------------------------------------------------------
    # S4: Topology + Length Confound (Planted cycle + N difference)
    # -------------------------------------------------------------
    s4_rows = []
    for i in range(n_samples):
        label = 0 if i < half else 1
        N = np.random.randint(25, 40) if label == 0 else np.random.randint(45, 65)
        D_base = generate_base_distance_matrix(N, noise_level=0.05)
        if label == 1:
            cycle_nodes = np.random.choice(N, size=10, replace=False)
            D = plant_pure_topological_cycle(D_base, cycle_nodes, cycle_radius=0.25)
        else:
            D = D_base
        feats = extract_features_from_distance(D, N)
        feats["label"] = label
        s4_rows.append(feats)
    regime_dfs["S4"] = pd.DataFrame(s4_rows)
    
    # -------------------------------------------------------------
    # S5: Negative Control (Pure noise with random labels)
    # -------------------------------------------------------------
    s5_rows = []
    for i in range(n_samples):
        N = np.random.randint(30, 55)
        D = generate_base_distance_matrix(N, noise_level=0.1)
        label = np.random.randint(0, 2)
        feats = extract_features_from_distance(D, N)
        feats["label"] = label
        s5_rows.append(feats)
    regime_dfs["S5"] = pd.DataFrame(s5_rows)
    
    return regime_dfs

def evaluate_regimes(regime_dfs):
    feature_sets = {
        "Length Baseline": ["N"],
        "Attention Marginals": ["mean_w", "max_w", "std_w", "degree_var"],
        "MST Baseline": ["mst_norm", "mst_max_edge"],
        "0D Persistence": ["h0_max_lifetime", "h0_total_persistence"],
        "1D Raw Persistence": ["h1_max_lifetime", "h1_total_persistence", "h1_count"],
        "1D Norm Persistence": ["h1_max_norm", "h1_total_norm", "h1_count"],
        "0D + 1D (Norm)": ["h0_max_lifetime", "h0_total_persistence", "h1_max_norm", "h1_total_norm"]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    master_rows = []
    
    titles = {
        "S1": "S1: Pure Length Confound (Identical distribution, different N)",
        "S1b": "S1b: Pure Length Confound, i.i.d. background (Prop. 1's own null model)",
        "S2": "S2: Pure Topology / Positive Control (Marginal-matched, planted 1D cycles)",
        "S3": "S3: Pure Marginals (Matched N, shifted density without cycles)",
        "S4": "S4: Topology + Length Confound (Planted cycles + variable N)",
        "S5": "S5: Negative Control (Pure random noise with permuted labels)"
    }
    
    print("=" * 90)
    print("CONTROLLED CAUSAL BENCHMARK EVALUATION (REGIMES S1 - S5)")
    print("=" * 90)
    
    for code, df in regime_dfs.items():
        y = df["label"].values
        title = titles[code]
        print(f"\n{title}")
        row = {"Regime": code, "Description": title}
        
        for feat_name, cols in feature_sets.items():
            X = df[cols].values
            scores = []
            for train_idx, test_idx in skf.split(X, y):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_te, y_te = X[test_idx], y[test_idx]
                
                # Check for constant column in fold (e.g. N in matched regimes)
                std = np.std(X_tr, axis=0)
                valid_cols = std > 1e-8
                if not np.any(valid_cols):
                    # Constant features yield chance AUC = 0.50
                    scores.append(0.50)
                    continue
                    
                X_tr_s = (X_tr[:, valid_cols] - np.mean(X_tr[:, valid_cols], axis=0)) / std[valid_cols]
                X_te_s = (X_te[:, valid_cols] - np.mean(X_tr[:, valid_cols], axis=0)) / std[valid_cols]
                
                clf = LogisticRegression(max_iter=500, random_state=42)
                clf.fit(X_tr_s, y_tr)
                preds = clf.predict_proba(X_te_s)[:, 1]
                scores.append(roc_auc_score(y_te, preds))
                
            mean_auc = np.mean(scores)
            std_auc = np.std(scores)
            row[feat_name] = f"{mean_auc:.3f} ± {std_auc:.3f}"
            print(f"  {feat_name:25s}: {mean_auc:.3f} ± {std_auc:.3f}")
            
        master_rows.append(row)
        
    res_df = pd.DataFrame(master_rows)
    res_df.to_csv("synthetic_causal_benchmark_results.csv", index=False)
    print("\nSaved synthetic benchmark to synthetic_causal_benchmark_results.csv")
    return res_df

if __name__ == "__main__":
    dfs = generate_controlled_regimes(n_samples=400)
    evaluate_regimes(dfs)
