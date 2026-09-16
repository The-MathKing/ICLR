"""
compute_complexity_profile.py
=============================
Computes asymptotic computational complexity and empirical wall-clock / memory benchmarks
for each stage of the topological hallucination detection pipeline across sequence lengths.
"""

import time
import tracemalloc
import numpy as np
import scipy.sparse.csgraph as csgraph
import ripser
import pandas as pd
from sklearn.linear_model import LogisticRegression

def benchmark_pipeline_complexity():
    seq_lengths = [32, 64, 128, 256, 512]
    n_layers = 32
    results = []
    
    print("=" * 85)
    print("COMPUTATIONAL COMPLEXITY & RUNTIME PROFILING BENCHMARK")
    print("=" * 85)
    
    for N in seq_lengths:
        # 1. Symmetrization benchmark
        A = np.random.rand(n_layers, N, N).astype(np.float32)
        
        tracemalloc.start()
        t0 = time.perf_counter()
        W = np.maximum(A, np.transpose(A, (0, 2, 1)))
        D = 1.0 - W
        for l in range(n_layers):
            np.fill_diagonal(D[l], 0.0)
        t_sym = (time.perf_counter() - t0) * 1000.0  # ms
        mem_sym = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        
        # 2. MST computation (all layers)
        tracemalloc.start()
        t0 = time.perf_counter()
        for l in range(n_layers):
            mst = csgraph.minimum_spanning_tree(D[l])
            _ = mst.sum()
        t_mst = (time.perf_counter() - t0) * 1000.0
        mem_mst = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        
        # 3. 0D Persistent Homology (all layers)
        tracemalloc.start()
        t0 = time.perf_counter()
        for l in range(n_layers):
            _ = ripser.ripser(D[l], maxdim=0, distance_matrix=True)
        t_h0 = (time.perf_counter() - t0) * 1000.0
        mem_h0 = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        
        # 4. 1D Persistent Homology (all layers)
        tracemalloc.start()
        t0 = time.perf_counter()
        for l in range(n_layers):
            _ = ripser.ripser(D[l], maxdim=1, distance_matrix=True)
        t_h1 = (time.perf_counter() - t0) * 1000.0
        mem_h1 = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        
        results.append({
            "Seq Length (N)": N,
            "Symmetrization (ms)": f"{t_sym:.2f}",
            "MST All Layers (ms)": f"{t_mst:.2f}",
            "0D Ripser (ms)": f"{t_h0:.2f}",
            "1D Ripser (ms)": f"{t_h1:.2f}",
            "1D Ripser Peak Mem (MB)": f"{mem_h1:.2f}",
            "MST/0D Speedup over 1D": f"{(t_h1 / max(1e-3, t_mst)):.1f}x"
        })
        
        print(f"N = {N:3d} | Symmetrize: {t_sym:5.2f}ms | MST: {t_mst:6.2f}ms | 0D: {t_h0:6.2f}ms | 1D: {t_h1:7.2f}ms | 1D Memory: {mem_h1:5.2f}MB")
        
    df = pd.DataFrame(results)
    df.to_csv("computational_complexity_profile.csv", index=False)
    print("\nComplexity profile saved to computational_complexity_profile.csv")
    return df

if __name__ == "__main__":
    benchmark_pipeline_complexity()
