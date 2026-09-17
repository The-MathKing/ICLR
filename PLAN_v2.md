# PLAN v2 — from "audit" to "The Topology Is the Sink"

**Written:** 2026-09-16 · **Deadline:** 2026-09-19 · **Hardware here:** M-series Mac, 16 GB, 10 cores (MPS).
**Off-box:** RTX 5080 + 32 GB (only needed for Mistral-7B; see §6).

---

## 1. Why the current paper is not novel enough

Everything the current draft proves is either classical (0D PH = MST, Gower & Ross 1969)
or near-trivial (length normalization, directed-flag collapse). Its empirical story, "1D adds
~nothing," is a *negative* result with no mechanism. Reviewers will score that 3–5.

## 2. The new thesis (one sentence)

> **In decoder-only LMs, persistent homology of attention graphs is, provably and empirically,
> a measurement of the attention sink:** 0D persistence equals total sink attention (the
> "star" around BOS) up to a computable defect δ, and every higher-dimensional bar has
> lifetime ≤ δ. The "topological" hallucination signal is a sink signal, and what is left
> after removing the sink can be measured directly.

This connects two literatures that do not cite each other: attention-graph TDA
(Kushnareva, TOHA, HalluZig, CHARM) and attention sinks (Xiao et al. 2023; Gu et al. 2024;
Sun et al. 2024 massive activations). It explains *all* of the draft's loose ends:
- Mistral's 99.99% zero 1D rate → exact coning (δ = 0).
- ρ = 0.90 between P̄₀/N and the 1D zero-rate → both are functions of sink mass.
- Sub-linear α̂ on real attention vs Ω(N) under i.i.d. → sinks cone away the cycles.
- H3's strong correlation with peak row-attention → the row max is usually the sink.
- Prompt sensitivity of Mistral's degeneracy → a system turn redistributes sink mass.

## 3. New theory (all verified numerically before writing)

**Theorem A (Sink-coning reduction).** Let D be a symmetric dissimilarity on [N], s a vertex, and
δ_s = max_{u≠v, u,v≠s} (max(D_su, D_sv) − D_uv)⁺ the *coning defect*.
1. (Exact) If δ_s = 0: every VR sublevel set is a cone over s plus isolated vertices, so
   H_k = 0 for all k ≥ 1 at every scale; the star at s is an MST; P₀ = Σ_u D_su and
   max 0D lifetime = max_u D_su.
2. (Stable) In general, every H_k bar (k ≥ 1) has lifetime ≤ δ_s, and
   Σ_u D_su − (N−1)δ_s ≤ P₀ ≤ Σ_u D_su.
   *Proof:* D' = max(D, cone(s)) satisfies D ≤ D' ≤ D + δ_s and δ'_s = 0; the inclusions
   K^D_b ⊆ K^{D'}_{b+δ} ⊆ K^D_{b+δ} factor the persistence map through H_k = 0.
   MST weight is (N−1)-Lipschitz in sup-norm; the star is a spanning tree.
3. (Causal attention) with s = 0 (BOS): D_0u = 1 − A_u0 and
   δ_0 = max_{1≤v<u} (A_uv − min(A_u0, A_v0))⁺.
   So P₀ ≈ (N−1) − (total sink attention), i.e. **0D TDA = sink mass**.

Sanity check done: 3,000 random causal matrices, 0 violations, max observed lifetime/δ = 0.99.

**Proposition B (Ω(N) under i.i.d. weights; replaces Conjecture 1).** P₁ = ∫β₁(K_ε)dε,
K_ε = flag complex of G(N, ε); the strong Morse inequality β₁ ≥ f₁ − f₀ − f₂ integrated over
ε ∈ [4/N, N^{-1/2}] gives E[P₁(N)] ≥ N/4 − √N − N/24 − o(N).

**Theorem C (directed collapse)**: kept, statement fixed.
**Fact 1 (0D = MST)**: demoted from theorem.

## 4. New experiments

| ID | What | Where | Est. time |
|---|---|---|---|
| X0 | `sinktda/` library: 0D/1D (full, sink-deflated, answer-only), sink stats, δ, per-head 0D + sink mass + entropy, Lookback ratio, LLM-Check attention score, log-prob stats, hidden states | code | 1 h |
| X1 | Re-extract TruthfulQA (817 q × 2) for Qwen2.5-3B, Phi-3, TinyLlama, SmolLM (native templates; must reproduce old 0D exactly) | Mac | ~2 h |
| X2 | Re-extract HaluEval QA (Qwen2.5-3B 2,000 q; Qwen2.5-1.5B 500 q) | Mac | ~2 h |
| X3 | **On-policy TriviaQA**: 2,000 validation questions, greedy answers from each model, alias-match grading, attention on the model's own answer (Qwen2.5-3B, Qwen2.5-1.5B, Phi-3, TinyLlama) | Mac | ~2–3 h |
| X4 | Mistral-7B TruthfulQA (native + chat template) with the same extractor | **5080** | ~30 min |
| X5 | Theory checks on real attention: δ distributions, bound checks on every (example, layer), fraction exactly coned, δ vs 1D zero-rate | CPU | minutes |
| X6 | Synthetic: Prop. B lower bound vs simulation; sink-strength dose-response (1D → 0 as sink grows); planted-cycle dose-response inside real attention | CPU | 30 min |
| X7 | Unified timing profile on real attention (single protocol) | CPU | 10 min |

## 5. Analyses (one canonical evaluator, `sinktda/evaluate.py`)

Protocol: StandardScaler + L2-LR; grouped 10-fold; **5 repeated seeds**, OOF predictions
averaged across seeds; 10k group-cluster bootstrap on the seed-averaged predictions; TOST with
the 90% interval; **TOST power** (not superiority power); within-pair accuracy on paired benchmarks.

Primary tests, per setting:
- **T1 (the reduction):** SINK (2/layer: star weight, max star edge) vs 0D (2/layer). Equivalence expected.
- **T2:** 0D + SINK vs SINK: does topology add anything beyond the sink?
- **T3:** SINK + deflated(0D+1D) vs SINK: is there non-sink topological signal?
- **T4:** 0D + 1D vs 0D (the draft's H2, now at full N).
- **T5:** the baseline table: length, lexical (TF-IDF), log-prob, LLM-Check, Lookback Lens,
  hidden-state probe, per-head sink vs per-head 0D vs per-head entropy.
- **T6:** answer-only subgraph (no sink inside) 0D/1D.

## 6. Mistral-7B on the 5080

```bash
python -m sinktda.extract --setting truthfulqa --model mistralai/Mistral-7B-Instruct-v0.2 --template native
python -m sinktda.extract --setting truthfulqa --model mistralai/Mistral-7B-Instruct-v0.2 --template chat
```
Copy the `sinktda_out/` files back; every analysis picks them up automatically.

## 7. Paper rewrite (9 pages main text)

New title: *The Topology Is the Sink: Persistent Homology of Causal Attention Graphs Measures Attention Sinks*.
1. Intro (sink thesis, 4 contributions) · 2. Related work (TDA, sinks, hallucination, benchmark artifacts)
3. Theory (Fact 1, Thm A, Prop B, Thm C) · 4. Setup (benchmarks incl. on-policy; protocol)
5. Results: 5.1 the reduction holds on real attention (δ, bounds, T1); 5.2 what remains after the sink
(T2, T3, T6); 5.3 does it detect hallucinations? baselines + on-policy (T5); 5.4 1D (T4 + Prop B + coning)
6. Benchmark confounds (short) · 7. Discussion, limitations.
Appendices: proofs, full tables, legacy audit material, synthetic, timing.
Remove all revision-history language; cite HalluZig; verify 2026 references.

## 8. Order of execution

X0 → start X1 in background → X5/X6 while it runs → X2 → X3 → evaluator → tables/figures → paper.
