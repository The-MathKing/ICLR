# Revitalization plan: raising Contribution from 2 to 3

Written 2026-09-16 (deadline 2026-09-19). This memo was written before the large changes and is updated with outcomes at the end (§7).

## 0. Diagnosis

The AC's Contribution 2 has one cause. Every new statement in the paper concerns a *construction* (the head-averaged Vietoris–Rips barcode of the whole attention graph) that no current state-of-the-art detector uses. The theorem also combines textbook ingredients. A reviewer can accept all of it and still ask "so what?".

The fastest route to "new and consequential" is to aim the same machinery at a published, well-performing topological detector and prove what it computes. TOHA (Bazarova et al., ACL 2026) is the obvious target:
- it is SOTA-claiming;
- it is training-light;
- its score is a 0D quantity (the MSF of response tokens attached to a contracted prompt), so our cone argument applies to it *exactly*;
- its own paper reports that "attention to ⟨s⟩" is a weak feature (Table 9: 0.61–0.65 vs. 0.86–0.98 AUC for MTop-Div), and SinkProbe (Binkowski et al. 2026) only remarks qualitatively that sink nodes "are likely to appear in the MST".

So an exact identity between TOHA and a one-line first-order attention statistic is new, falsifiable, and directly contradicts a published claim if it holds empirically.

## 1. Ranking of the candidate directions

Score = (expected contribution gain) × (probability it works) ÷ (hours).

| # | Direction | gain | P(works) | hours | decision |
|---|---|---|---|---|---|
| 1 | Reduce TOHA (and position HalluZig) | high: moves the paper from "a construction nobody uses" to "the published SOTA topological detector" | 0.7. The identity is a theorem and was unit-tested against ripser: 400 random cases, error 2.6e-8, 0 sandwich violations. The empirical question is whether TOHA's *selected* heads are coned. | 8 | **do (A)** |
| 2 | Full-diagram / vectorization corollary | medium: "every stable vectorization is within Lδ of a function of the sink column" | 0.95 | 2 | **do (B)**, stated as a corollary |
| 3 | δ₀ / per-head δ₀ as a detector | low to medium | 0.3 | 1 (already running) | **do (C)**, one sentence either way |
| 4 | Precision fragility (fp32, int8) | medium-low: practical, not conceptual | 0.6 | 3–4 | defer; the existing fp16/bf16 table stays in the appendix |
| 5 | Causal sink intervention | medium | 0.4: changing the sink changes the model | 4–6 | partial: `toha.py --sink-bias` is implemented; run only if time permits |
| 6 | ≥7B scale | high for reviewers | needs the 5080 | 0.5 (scripting) | **scripts only**; results pending |

HalluZig is **not** reducible by the cone theorem as published. It uses graph (1-skeleton) homology of top-percentile edge sets and zigzags G_l → G_l ∪ G_{l+1} ← G_{l+1}; a cone *graph* has cycles, because its triangles are not filled. We state this honestly in the related work and limitations. Reimplementing zigzag persistence (no dionysus/fzz installed) is not worth the risk this week.

## 2. What each chosen direction adds to the contribution statement

- **A (TOHA):** "TOHA's topological divergence equals, exactly on prompt-coned heads and within δ_P otherwise, the mean over response tokens of one minus their largest prompt attention. In our settings, TOHA's selection algorithm run on this first-order statistic, or on response-restricted sink attention, reproduces TOHA's AUC."
- **B (corollary):** "On a graph coned at s, the whole VR barcode is the star diagram {[0, D_su)}; in general, it lies within bottleneck distance δ_s of the star diagram. So persistence images, landscapes, entropy and Betti curves are all (Lipschitz-)functions of the sink column up to δ_s."
- **C (δ₀ detector):** either "the coning defect is itself a signal beyond sinks and probes", or one sentence reporting the null.

## 3. Hours, risks and fallbacks

| | hours | main risk | if it fails, we report |
|---|---|---|---|
| A | 1.5 extraction (11 settings × 2–10 min, no ripser), 1 eval, 3 writing | TOHA's selected heads are copy heads with π_u ≠ A_u0, or δ_P > 0 | The identity to max-prompt attention still holds (theorem + 0 violations). The sink-specific claim is weakened to "copy/sink heads", with the measured share of selected heads that are coned. |
| B | 2 writing | the statement is folklore | cite stability; present it as a corollary, not a contribution |
| C | done | null | one sentence in §5.2 |

## 4. Exact new experiments

Run with `HF_HOME=/Volumes/2TB/hf_cache TMPDIR=JOBLIB_TEMP_FOLDER=/Volumes/2TB/iclr/.tmp SINKTDA_JOBS=2`, one heavy job at a time (`sinktda/run_toha.sh`).

```bash
# C: coning defect as a detector (CPU only, ~2 GB)
python -m sinktda.defect_probe
# A: per-head TOHA quantities, one forward pass per row, bf16, no ripser
#    (peak ≈ model weights + one layer's attention rows: ≤ 9 GB for Phi-3)
python -m sinktda.toha extract --bench {truthfulqa,halueval,triviaqa} --model {qwen3b,qwen1.5b,phi3,tinyllama,smollm} [--n ...]
python -m sinktda.toha evaluate          # CPU, ~2 GB
# optional causal probe
python -m sinktda.toha extract --bench truthfulqa --model qwen1.5b --sink-bias 4 --tag sb4
```

Outputs: `sinktda_out/<setting>/toha.npz` and `sinktda_results/toha_{checks,auc,comp}.csv`. `report.py` turns them into macros, `\SinkTableToha`, and the new §5.x.

**Evaluation of TOHA.** We run Algorithm 1 of Bazarova et al. faithfully:
1. Rank heads by Δ = mean over hallucinated − mean over grounded on the training fold.
2. Choose N_opt ≤ 10 by training AUROC.
3. Score a test example by the mean over the selected heads.

This runs inside the same grouped 10-fold × 3-seed CV as everything else, and the identical algorithm is applied to `maxp` and `sinkr`. We also fit a supervised all-heads probe (their Table 9 setting). Tests:
- TOHA − first-order under TOST at ±0.015;
- the stacked increment of TOHA over its first-order counterpart;
- the stacked increment of TOHA over NONTOPO.

## 5. Space plan (9 pages)

- **New:** Proposition 3 (TOHA) with a proof sketch (~0.35 page), a TOHA table plus paragraph (~0.45 page), and Corollary 2 (vectorizations; ~0.15 page).
- **Moves to the appendix:**
  - the directed-flag paragraph (keep one sentence in the main text);
  - the "What i.i.d. weights would give" paragraph and Proposition 1 (keep a two-sentence pointer; Fig. 2 left panel stays);
  - the "Benchmark artifacts" subsection (merged into one paragraph of §5.4);
  - the per-head reduction paragraph (one sentence remains).
- **Shortened:** the related work, and the forest-figure caption.

## 6. Draft framing (decided after A's numbers; see §7)

- **Title (if A holds):** *Topological Hallucination Detectors Measure Attention Sinks*.
- **Title (if A only half-holds):** keep *The Topology Is the Sink*, with TOHA as the second headline.
- **Draft contributions:**
  1. A reduction theorem for Vietoris–Rips barcodes of any dissimilarity with an approximately dominating vertex, with a bottleneck bound to the star diagram that covers all standard vectorizations.
  2. An exact reduction of TOHA's topological divergence to a first-order attention statistic, verified on N graphs with 0 violations. TOHA's own head-selection algorithm on the first-order statistic matches TOHA within ±0.015 AUC in k/11 settings.
  3. Verification that PH-based features on 568k real graphs reduce to sink attention.
  4. A detection audit with strong baselines and an on-policy benchmark.

## 7. Outcomes (2026-09-17)

| Direction | Status | Result |
|---|---|---|
| A: TOHA reduction | **worked** | 12.3M head graphs, 0 sandwich violations. 57–88% are exactly prompt-coned, with identity to 6·10⁻⁸. Median per-head ρ(d, π̄) is 0.954–0.998. Stacking d onto π̄ adds 0.000–0.007 (equivalent 11/11). Supervised π̄ vs d is equivalent 11/11; supervised response-sink vs d is equivalent 9/11 (the misses are SmolLM, which has no sink, and one wide interval). |
| A, caveats | reported | TOHA's discrete selection on π̄ gives −0.012 to +0.018 (3 better, 2 worse, non-inferior 9/11). On the sink score it loses up to 0.049 in sink models: the selected heads often have a non-sink top prompt token. TOHA adds up to 0.013 over NONTOPO (3/11 significant). |
| B: whole-barcode corollary | **done** | Corollary 1 with proof; numerical check 0 violations (`check_theory.py`). |
| C: δ₀ detector | **partial positive** | Per-head δ₀ AUC 0.69–0.96; +0.006 to +0.027 over per-head sink (11/11 significant); 0.000 to +0.008 over NONTOPO (equivalent 11/11). |
| 5: causal sink bias | **worked at the graph level** | b=−2 lowers the prompt-coned share, the sink-top share and ρ(d, sink score) (Qwen 0.89→0.61, TinyLlama 0.97→0.41); b=+4 raises ρ (0.96, 0.996). TOHA's AUC barely moves, and the AUC gap is not monotone (reported). The first run was invalid (non-causal custom attention), was discarded and fixed, and is now guarded. |
| 4: fp32/int8 | deferred | the existing fp16/bf16 table stays |
| 6: ≥7B | scripted | `RUN_ON_5080.md` (Mistral-7B, Qwen2.5-7B, Llama-3.1-8B) |

**Framing decision.** The title keeps the brand and states the new headline: *The Topology Is the Sink: Topological Hallucination Detectors Reduce to First-Order Attention Statistics*. "Measure attention sinks" was rejected as an over-claim: TOHA reduces to *max-prompt* attention, which equals the sink only when the sink is the top prompt token.

**Space.** Proposition (i.i.d.) and §5.3 (1D) moved to App. A/D, and the forest figure to App. E. The artifacts subsection became a paragraph. The main text ends on p. 9.
