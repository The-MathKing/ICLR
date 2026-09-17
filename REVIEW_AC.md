# Area-Chair Review — "The Topology Is the Sink: Persistent Homology of Causal Attention Graphs Measures Attention Sinks"

Reviewed: `paper/paper.tex` + `appendix_sink.tex` + generated inputs (state of the working tree on 2026-09-16), code in `sinktda/`, results in `sinktda_results/` and `sinktda_out/`. All numbers quoted below were recomputed from those files unless stated otherwise.

---

## 1. Summary

The paper argues that Vietoris–Rips persistence of head-averaged, symmetrized causal attention graphs (the input to TDA hallucination detectors) is largely determined by the attention sink at token 0. It introduces a "coning defect" δ_s, bounds higher-dimensional bars and 0D total persistence in terms of δ_s and the star weight, checks these bounds on 568k real layer graphs, and audits detection on TruthfulQA, HaluEval and a new on-policy TriviaQA benchmark against non-topological baselines.

**Primary contribution type:** an empirical/analytical audit (negative result) supported by elementary theory.

## 2. Scores (pre-fix)

| | Score | Justification |
|---|---|---|
| Soundness | **2 (fair)** | The core bounds are correct, but the headline characterization of coning is misstated, the P₀–sink correlation is length-confounded, the i.i.d. simulation does not match Proposition 1's model, and the "0D adds nothing beyond non-topological probes" equivalences come from an early-fusion design that makes them nearly automatic. |
| Presentation | **3 (good)** | Clear writing and a well-macroed pipeline, but the compiled PDF is not in Times (style violation), there is a notation clash (P1–P4 vs P₀/P₁), and several "every/all" statements don't match the tables. |
| Contribution | **2 (fair)** | The sink↔TDA connection is useful and timely, but the theorem is a direct combination of classical cone/dominated-vertex and stability arguments, Proposition 1's Θ(N) order is already known (Hiraoka–Shirai 2017; Hino–Kanazawa 2019), and a 2026 paper (SinkProbe) already ties sinks to hallucination detection. |
| Overall | **5 (marginally below the acceptance threshold)** | A useful, honest audit whose theory is over-claimed in novelty and whose key empirical claims need a length control and a proper incremental-value test. |
| Confidence | **4** | I checked every proof step, re-derived the asymptotics numerically, and traced all macros to CSVs. I did not rerun model extractions before writing this review. |

## 3. Strengths

1. **A clean, checkable bridge between two literatures.** Corollary 1 gives δ₀ and S₀ in closed form for causal attention, and Table 1 shows 0 bound violations across 568,028 graphs with exact coning in 36–88% of graphs of sink models. Also, whenever a graph is coned at any vertex, it is (up to 14 graphs in 72k) coned at token 0 (`frac_cells_coned_any_apex` ≈ `frac_cells_coned`, `summary_theory.csv`).
2. **Unusually disciplined evaluation code.** There is one evaluator (`sinktda/evaluate.py`), question-grouped CV, TF-IDF fitted within folds, and every prose number is macro-generated (`report.py:write_numbers`). I found no train/test leakage.
3. **An on-policy benchmark plus lexical controls.** Together they expose that teacher-forced conclusions (log-prob weak, 0D stronger) flip on-policy (Table 2).
4. **Useful mechanistic evidence.** The label-free layer split (coned layers add −0.001…+0.004 beyond Sink; open layers +0.003…+0.029) and the synthetic sink dose-response are good mechanistic evidence.

## 4. Weaknesses

### Theory

**W-T1 (major) — The characterization of exact coning is wrong in the abstract and Corollary 1.**
- *Location:* `paper.tex:56` ("δ₀=0 exactly when no token attends to another more than both attend to the sink"); `paper.tex:119` ("…more than both of them attend to the sink").
- *Evidence:* δ₀>0 iff A_uv > **min**(A_u0, A_v0), i.e. more than *either* attends to the sink. Counterexample: A₁₀=.30, A₂₀=.50, A₂₁=.35. Token 2 attends to token 1 less than token 2 attends to the sink, yet `delta_bos_causal` = 0.05 > 0.
- *Fix:* Replace "both" with "either" (u attends to v more than u, or v, attends to the sink).

**W-T2 (major) — Proposition 1's growth order is known, and its numerical check uses the wrong null model.**
- *Location:* `paper.tex:70,130–137`; `appendix_sink.tex:41–48`; `synthetic_causal_benchmark.py:46–55`.
- *Evidence (known result):* For the random clique-complex process, E[L₁] = Θ(n) is Theorem 1.5 of Hino & Kanazawa (J. Math. Soc. Japan 71(3), 2019), which tightens the upper bound; Hiraoka & Shirai (RSA 2017) had already given the linear lower bound. Neither is cited. The "≥3N/8" constant is a new but elementary addition. I verified that the integral equals 3N/8 − √(3N) + O(1): (3N/8 − bound)/√N = 1.62, 1.67, 1.70, 1.72 at N = 256, 1k, 4k, 16k.
- *Evidence (wrong null model):* The simulation behind Fig. 2 (left) uses D = (U+Uᵀ)/2. That has a triangular distribution, not Uniform[0,1], so the plotted mean is not E[P₁] under Proposition 1's hypothesis.
- *Fix:* Cite both papers and present Proposition 1 as an explicit-constant, elementary version of a known Θ(N) law; note that the upper order is also linear. Re-simulate with i.i.d. uniform upper-triangular weights.

**W-T3 (minor) — The bound in Theorem 1(a) is attained, but the paper says otherwise.**
- *Location:* `paper.tex:111` ("tight up to the constant") and `paper.tex:119` ("shorter than δ₀").
- *Evidence:* A 4-cycle at distance .5, with the apex and diagonals at 1, has δ₀ = .5 and an H₁ bar [.5, 1) of length exactly δ₀.
- *Fix:* Say "attained" and "at most δ₀", and give the example.

**W-T4 (minor) — The cycle-property step does not handle ties.**
- *Location:* `appendix_sink.tex:30`.
- *Evidence:* "every non-star edge … can be excluded" applies the cycle property to all edges at once. With ties (common, since D'_uv = D'_su is typical) this needs an argument.
- *Fix:* Use the rooted-tree argument. Root any spanning tree T at s; the parent edge of u has D'-weight ≥ D'_su, so w_{D'}(T) ≥ S_s. This also gives the bottleneck bound.

**W-T5 (minor) — The distance-choice remark is false for merely non-decreasing transforms.**
- *Location:* `paper.tex:122–124`.
- *Evidence:* A non-decreasing (not strictly increasing) transform can create coning. For D₀₁ = D₀₂ = .6, D₁₂ = .5, clamping at .5 changes δ₀ from 0.1 to 0.
- *Fix:* Require a strictly increasing transform with f(0)=0.

**W-T6 (minor) — Novelty and naming of the theory.**
- *Location:* `paper.tex:64,68,97–109,139`; `appendix_sink.tex:52–59`.
- *Evidence:* Lemma 1 is the classical fact that a flag complex with a dominating vertex is a cone; see strong collapses (Barmak–Minian 2012) and edge collapse of flag filtrations (Boissonnat–Pritam 2020). Step 2 is the standard one-sided interleaving/stability argument (Cohen-Steiner et al. 2007; Chazal et al. 2009); `cohensteiner2007stability` and `chazal2009proximity` are in the bib but never cited. The "directed flag collapse theorem" is the observation that the directed flag complex of a transitive tournament is the ordered simplex. Its hypothesis A_ij > 0 is unnecessary, and "directional treatments … do not transfer to decoders" over-reaches.
- *Fix:* Cite these works and position the contribution as the quantitative specialization plus the closed form. Rename the directed-flag theorem to a Proposition, drop the positivity hypothesis, and soften the transfer claim to "flag-complex-based directional treatments".

**W-T7 (minor) — Table 1 overstates what was checked.**
- *Location:* Table 1 caption; `paper.tex:148`; `evaluate.py:226–227`.
- *Evidence:* The caption says violations of "(a) or (b)" are counted, but the max-death half of (b) is never checked. Also, the text says tolerance 1e-5, while the P₀ check uses 1e-3.
- *Fix:* Add the max-death check (I ran it: 0 violations in all 11 settings) and state both tolerances.

### Empirical / methodology

**W-E1 (major) — The P₀–S₀ correlation is partly a length effect.**
- *Location:* Table 1 last column; `sink_results.tex:5,9`; Fig. 1 caption; abstract (ρ ≥ 0.999).
- *Evidence:* S₀ = (N−1)(1−m̄) and P₀ both scale with N; per layer, Spearman(S₀, N) = 0.96–0.999. After dividing both by (N−1), the sink settings keep median ρ ≥ 0.98, so the claim survives, but some layers drop to 0.04–0.08 (TQA Phi-3, Qwen-3B). SmolLM's ρ collapses from **0.967 to 0.41**. The Fig. 1 caption ("Even there, P₀ remains rank-aligned…") and the sentence "per-layer ρ drops to 0.967" therefore describe a length artifact. Also, "0D is, up to rank, the mean attention paid to the first token" holds only after this normalization.
- *Fix:* Report Spearman(P₀/(N−1), 1−m̄) in Table 1 and Fig. 1 (keep the raw version in the appendix), and rewrite the SmolLM sentences.

**W-E2 (major) — The "0D | non-topological" equivalences are close to guaranteed by design.**
- *Location:* Table 3 columns "0D|NT" and "Defl.|NT"; `sink_results.tex:31`; abstract.
- *Evidence:* The 72–216 topological columns are appended to a 1.9k–2.7k-dimensional early-fusion ridge model whose C is chosen from {.01, .1, 1}. The hidden-state block dominates, so any small block is swamped regardless of its information. TOST "power 1.00" is uninformative here.
- *Fix:* Add a late-fusion (stacked) test on the stored OOF scores: a grouped-CV logistic regression on [logit p_NT, logit p_X], compared against p_NT. Report it next to the early-fusion result. Soften "adds no measurable information" to what the stacked test supports.

**W-E3 (major) — Inference ignores training variability, and one reported number is wrong.**
- *Location:* `appendix_sink.tex:77`; `evaluate.py:140–159,185–208`.
- *Evidence:* Predictions are averaged over three CV seeds and then bootstrapped over questions. The intervals therefore treat the fitted predictor as fixed, which is anti-conservative for TOST. Separately, the appendix says the seed SD "is below 0.004 everywhere", but it reaches **0.008** (LEX, TrQA Phi-3) and exceeds 0.004 in five cells.
- *Fix:* State the limitation explicitly, use the `\MaxSeedSD` macro, and (nice-to-have) store per-seed OOF predictions and report the across-seed range of each key Δ.

**W-E4 (major) — dtype differs across settings and is misreported.**
- *Location:* `appendix_sink.tex:64` ("Models run in float16").
- *Evidence:* Per the logs, TQA Qwen-3B, Phi-3, TinyLlama, SmolLM and HE Qwen-3B ran in fp16, while TQA/HE Qwen-1.5B and all TriviaQA runs used bf16. The same fp16 Qwen-3B configuration produced garbage generations (accuracy 0.028, `logs/extract_all.log`). The `dtype` column is missing from four parquets.
- *Fix:* Re-extract the fp16 settings in bf16 (fits on 16 GB) and state the dtype per setting.

**W-E5 (major) — Tokenization-boundary bug on HaluEval.**
- *Location:* `extract.py:148`; `data.py:71`.
- *Evidence:* The prefix ends in "Answer: " (trailing space). Tokenized alone, it has more tokens than its span inside the full string, so `prompt_len ≥ seq_len` for 99/1000 (Qwen-1.5B) and 195/2000 (Qwen-3B) rows. In those rows the answer-only, log-prob, hidden-state and Lookback features are computed on a placeholder token. The affected rows are overwhelmingly short (right) answers, so the defect correlates with the label.
- *Fix:* Strip the trailing space from the prefix (and in general compute the boundary from the tokenized full string), then re-extract HaluEval.

**W-E6 (major) — On-policy labels are unvalidated string matches.**
- *Location:* `data.py:102–115`; `paper.tex:144,228`.
- *Evidence:* Alias containment mislabels paraphrases and hedged answers, and Janiak et al. (2025) show that lexical grading materially distorts hallucination-detection conclusions. There is no audit.
- *Fix:* Hand- or LLM-judge-audit a random sample of ≥200 answers per model and report the label-disagreement rate. At minimum, report refusal/hedge rates and state the limitation.

**W-E7 (minor) — The on-policy tokenization differs between generation and feature extraction.**
- *Location:* `extract.py:63` vs `extract.py:146`.
- *Evidence:* Generation uses `add_special_tokens=False`, but extraction uses the default. For TinyLlama this prepends a BOS token during extraction that the model never saw while generating.
- *Fix:* Disclose it now; unify the setting for future runs.

**W-E8 (major) — SmolLM is not a controlled "negative control".**
- *Location:* `sink_results.tex:9`; `paper.tex:69`; abstract.
- *Evidence:* SmolLM differs in weights, training data and template. "Never coned, as the theorem predicts" is tautological, since the theorem predicts nothing for non-coned graphs. The residual ρ = 0.967 is a length effect (W-E1).
- *Fix:* Call it a contrast rather than a control, drop "as the theorem predicts", and point to deflation (P4) as the within-model intervention.

**W-E9 (minor) — The 1D-scaling claims don't hold as stated.**
- *Location:* abstract ("its sub-linear growth on real attention is a signature of coning"); Table 5.
- *Evidence:* The HaluEval exponents are 1.41 and 1.10 (super-linear). The comparison is also against the (mis-simulated) i.i.d. model rather than random causal attention without a sink.
- *Fix:* Weaken the abstract to "slower growth than under i.i.d. weights", and add a random-causal-attention (b=0) reference exponent.

**W-E10 (minor) — The planted-cycle result is reported selectively.**
- *Location:* `sink_results.tex:25`.
- *Evidence:* At b=0 (no sink) H₁ also fails to detect the planted cycle (AUC ≤ 0.57), so "loops are informative in principle" is true only under a moderate sink.
- *Fix:* Report the non-monotonicity.

**W-E11 (minor) — Wording that the tables don't support.**
- *Location / evidence:*
  - "0D beats log-prob for four of five models" (`sink_results.tex:33`): the TinyLlama and SmolLM differences have 95% CIs covering 0, so only three are significant.
  - "attention graphs carry some non-sink geometry … on the on-policy benchmark" (`sink_results.tex:19`): only 1 of 4 on-policy Defl.|Sink intervals excludes 0.
  - "in any setting" (abstract): the NT tests exist for 10 of 11 settings.
  - "equivalent … in 9 of 9 settings" (per-head): these are the 9 sink settings with per-head features, and the Table 3 column adds only per-head total persistence (`PH_0DTOT`), not the `PH_0D` bank shown in Table 2.
- *Fix:* Qualify each statement and relabel the column.

**W-E12 (minor) — Protocol details are inaccurate or confounded.**
- *Location / evidence:*
  - `appendix_sink.tex:66`: the Qwen TruthfulQA prompt uses an explicit "You are a helpful assistant." system turn, not the default one.
  - HaluEval uses no chat template, while TruthfulQA/TriviaQA do, so cross-model and cross-benchmark comparisons mix templates.
  - The C-selection threshold (64 dims) puts the same bank in different regularization regimes across models (e.g., Sink has 72 dims for Qwen-3B but 56 for Qwen-1.5B).
- *Fix:* Correct the description and add one sentence on each confound.

**W-E13 (major, external) — Scale.**
- *Evidence:* The largest model has 3.8B parameters. Sink strength and coning shares are model-specific, and ICLR reviewers will expect at least one ≥7B model (Mistral-7B pending, `RUN_ON_5080.md`), ideally also Llama-3-8B.
- *Fix:* Run on the 5080.

### Novelty & framing

**W-N1 (major) — Missing and contradicted prior work.**
- *Location:* `paper.tex:62` ("These two literatures do not cite each other").
- *Evidence:* Binkowski, Adamczewski & Kajdanowicz (arXiv:2604.10697, Apr 2026; "SinkProbe") show that attention sinks drive hallucination detection and that prior attention-based detectors implicitly depend on them. The statement is therefore false, and SinkProbe is the closest related work.
- *Fix:* Cite and position against it. The contribution becomes the *topological* reduction and the audit of TDA detectors specifically. Also cite Clark et al. (2019) for BERT's sink-like attention (relevant because the original TDA pipelines used BERT), and Darcet et al. (2024).

**W-N2 (major) — The theory is over-claimed.** See W-T2 and W-T6. The contributions list presents three theorems as new; the introduction should say which parts are classical.

**W-N3 (minor) — Frieze's ζ(3) theorem is uncited.**
- *Evidence:* It gives E[P₀] → ζ(3) under i.i.d. weights and is the natural 0D counterpart to Proposition 1.
- *Fix:* Cite it next to Proposition 1.

### Presentation

**W-P1 (major) — The body font is Latin Modern, not Times.**
- *Location:* `paper.tex:2`; build log.
- *Evidence:* XeTeX/tectonic ignores `times` under TU encoding (`Font shape TU/ptm/m/n undefined`), so the body is set in Latin Modern. This violates the ICLR style and changes the page count.
- *Fix:* Load `fontenc` (T1) before `times`, or use fontspec with TeX Gyre Termes, then re-check pagination.

**W-P2 (minor) — Duplicate hyperref anchors.**
- *Location:* `paper.tex:7–18`.
- *Evidence:* hyperref is loaded before `float`, giving 13 "Object @table.N already defined" warnings; links can point to the wrong float.
- *Fix:* Load hyperref last.

**W-P3 (minor) — Notation clashes.**
- *Evidence:*
  - Predictions "P1–P4" clash with P₀/P₁ (total persistence).
  - ε is both the filtration scale and the TOST margin.
  - TOST is never expanded.
  - The Fig. 2 legend says "Prop. 2 lower bound" while the text calls it Proposition 1.
- *Fix:* Rename the predictions to (i)–(iv), use Δ_eq for the margin, expand TOST, and fix the legend.

**W-P4 (minor) — Figure 1 legend is illegible.**
- *Evidence:* The legend is 6.5 pt at 7 in width. The right panel also plots the length-confounded quantity (W-E1).
- *Fix:* Enlarge the legend and plot the normalized ρ.

**W-P5 (minor) — Stale README.**
- *Location:* `README.md`.
- *Evidence:* It still describes the superseded v1 paper ("Attention TDA Is a Minimum Spanning Tree", `master_pipeline.py`, int8 Mistral), while the reproducibility statement points to `sinktda/`. A reviewer following the README will reproduce the wrong paper.
- *Fix:* Rewrite it for the `sinktda` pipeline.

**W-P6 (minor) — Reproducibility details missing.**
- *Evidence:* The CV seeds (42–44), bootstrap seed, model revisions, library versions and per-setting dtype are not stated in the paper.
- *Fix:* Add them to Appendix B.

**Checked and fine:**
- The main text ends on page 8 (≤ 9).
- No undefined references and no overfull boxes.
- The author block is anonymous; the anonymous.4open.science link is fine.
- No names or paths appear in the `.tex` sources.

### Citation audit (`paper.bib`, cited entries only)

| key | status |
|---|---|
| chuang2024lookback | **WRONG.** The title conflates two titles, the venue is EMNLP 2024 (pp. 1419–1436), not ICML, and the authors are Chuang, Qiu, Hsieh, Krishna, Kim, Glass. |
| joshi2017triviaqa | **WRONG.** The listed authors are SpanBERT's. The correct title is "TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension", by Joshi, Choi, Weld, Zettlemoyer (ACL 2017, pp. 1601–1611). |
| samaga2026halluzig | **WRONG.** The first author is Shreyas N. Samaga, and the title is "HalluZig: Hallucination Detection using Zigzag Persistence" (EACL 2026). |
| flagser2020 | **WRONG venue.** It is Algorithms 13(1):19, 2020. |
| xiao2023efficient | Outdated: published at ICLR 2024. |
| janiak2025illusion | Verified (EMNLP 2025); upgrade from arXiv. |
| toha | Verified (ACL 2026, pp. 15449–15470). |
| charm | Verified (ICLR 2026). |
| hussain2026parallax | Verified (arXiv:2605.17028). |
| gower1969, farquhar2024semantic, lin2022truthfulqa, li2024halueval, azaria2023internal, manakul2023selfcheckgpt, chen2024inside, sriramanan2024llmcheck, kushnareva2021, cherniavskii2022, bauer2021ripser, kahle2009topology, edelsbrunner2010, gu2025attentionsink, sun2024massive, cancedda2024spectral, li2023iti, du2024haloscope, gururangan2018annotation, mccoy2019right, kossen2024semantic | Verified from knowledge (authors/venue/year consistent). |
| kostenok2023, perez2022topological, gardinazzi2024persistent, barbero2025first, burns2022discovering, marks2023geometry, orgad2024llms | Likely correct (arXiv IDs consistent); several now have venues (Burns: ICLR 2023; Marks: COLM 2024; Orgad: ICLR 2025). |
| **missing** | hino2019lifetime, hiraoka2017msa, frieze1985, barmak2012, boissonnat2020edge, chazal2014geometric (and cite the existing cohensteiner2007/chazal2009), binkowski2026sinkprobe, clark2019bert, darcet2024registers |

## 5. Questions for the authors

1. After normalizing by (N−1), which layers of Phi-3 and Qwen-3B have ρ(P₀, S₀) < 0.5, and are they the open layers?
2. Does 0D (or deflated PH) add anything to non-topological probes under late fusion, where the small block cannot be swamped by 2k hidden dimensions?
3. How often does string-match grading disagree with a human or LLM judge on your TriviaQA generations, and do the conclusions survive relabelling?
4. Do the fp16 and bf16 extractions of the same setting give the same coning rates and AUCs?
5. Does coning persist at 7–8B, and under Mistral's `[INST]` format versus its chat template?
6. How do your sink features compare to SinkProbe, which uses value-norm-weighted sinks?

## 6. Meta-review: what would move the score up one level (5 → 6)

- Correct the coning statement.
- Credit the known results (Hino–Kanazawa, Hiraoka–Shirai, cone/collapse, stability) and SinkProbe.
- Make the P₀–sink claim length-controlled.
- Replace or complement the early-fusion incremental tests with a late-fusion test.
- Fix the HaluEval boundary bug and the dtype heterogeneity.
- Fix the four wrong references and the font.

Reaching 8 additionally requires a ≥7B model and an audited on-policy label set (ideally a second on-policy dataset).

## 7. Fix list (ordered by severity)

1. `[text]` W-T1: "both" → "either" in the abstract and Corollary 1.
2. `[text]` W-N1: cite SinkProbe and remove "do not cite each other"; add Clark et al. and Darcet et al.
3. `[text]` W-T2/W-T6/W-N2/W-N3: cite Hino–Kanazawa, Hiraoka–Shirai, Frieze, Barmak–Minian, Boissonnat–Pritam, stability; reposition the theorems; rename the directed-flag theorem.
4. `[text]` Fix the four wrong bib entries and update the venues.
5. `[recompute]` W-E1: length-normalized ρ in `evaluate.theory_checks` → Table 1, Fig. 1, macros, and the SmolLM prose.
6. `[recompute]` W-E2: late-fusion (stacked) incremental tests from the OOF files → Table 3 and prose.
7. `[extract]` W-E4 + W-E5: fix the prefix boundary; re-extract the fp16 settings (TQA Qwen-3B, Phi-3, TinyLlama, SmolLM) and both HaluEval settings in bf16; re-evaluate; rerun the layer split.
8. `[recompute]` W-T2: re-simulate the i.i.d. model with true uniform weights; add a random-causal (b=0) reference exponent.
9. `[text]` W-E3: fix the seed-SD claim (macro) and state the fixed-predictor limitation.
10. `[text]` W-E6/W-E7: disclose grading and BOS limitations; `[recompute]` report refusal/hedge rates.
11. `[text]` W-E8, W-E9, W-E10, W-E11, W-E12: qualify the wording (SmolLM, scaling, planted cycle, "four of five", "any setting", per-head column, prompts, C threshold).
12. `[text]` W-T3, W-T4, W-T5, W-T7: tightness example, rooted-tree argument, strict monotonicity, max-death check and tolerances (`[recompute]` for the check).
13. `[text]` W-P1, W-P2, W-P3, W-P4: font, hyperref order, notation, figure legends.
14. `[text]` W-P5, W-P6: rewrite the README; add seeds, versions and dtype to the appendix.
15. `[external]` Mistral-7B (and ideally Llama-3-8B) on the RTX 5080 (`sinktda/RUN_ON_5080.md`).
16. `[external]` A human or LLM-judge audit of ≥200 on-policy labels per model.
17. `[external]` Anonymity: the public GitHub repo and third-party PDFs in `refrences/`; make sure only the anonymized mirror is linked.
18. `[external]` Swap in the correct ICLR-year style file if the 2026 one is not current.
19. `[external]` Final verification of the remaining 2025–2026 citations against publisher pages.
20. `[external]`, nice-to-have: original TOHA/HalluZig code as baselines; semantic entropy (needs sampling); a second on-policy dataset; per-head deflation; a SinkProbe comparison; per-seed OOF storage for training-variance-aware intervals.

---

## 8. Post-fix assessment (2026-09-16, after Phase 2)

**Build check.** The paper compiles with tectonic: 29 pages, no errors, no overfull boxes, no undefined references, and no duplicate hyperref anchors. The main text (Sections 1–6, including Limitations) ends on page 9; the reproducibility and LLM-usage statements follow on the same page. The body font is now Times (NimbusRomNo9L is embedded). I read every main-text page, plus the new appendix pages, as rendered images.

### What was fixed

| # | Item | Change | Result |
|---|---|---|---|
| 1 | W-T1 | Abstract and Corollary 1 now say "more than **either**" and give the exact condition A_uv ≤ A_u0 and A_uv ≤ A_v0. | fixed |
| 2 | W-N1 | SinkProbe (Binkowski et al. 2026), Clark et al. 2019 and Darcet et al. 2024 are cited, and "do not cite each other" is removed. | fixed |
| 3 | W-T2/T6/N2/N3 | Hino–Kanazawa, Hiraoka–Shirai, Frieze, Barmak–Minian, Boissonnat–Pritam, Chazal et al. 2009/2014 and Cohen-Steiner et al. are cited. Contribution 1 is reframed as combining classical ingredients. Proposition 1 is stated as an explicit-constant version of a known Θ(N) law (= 3N/8 − √(3N) + O(1)); the trivial upper bound is removed. The directed-flag "theorem" is now a proposition, without the positivity hypothesis and with the transfer claim softened. An appendix paragraph "Relation to known results" is added. | fixed |
| 4 | Bib | Lookback Lens, TriviaQA, HalluZig and flagser are corrected; Xiao et al. is updated to ICLR 2024 and Janiak et al. to EMNLP 2025; nine new entries are added (verified via web). | fixed |
| 5 | W-E1 | `evaluate.theory_checks` adds the length-normalized ρ (P₀/(N−1) vs 1−m̄). Table 1 shows raw and normalized columns; Fig. 1 plots the normalized one. The abstract now uses the normalized minimum (≥ **0.98**). SmolLM is reported honestly: raw 0.967, normalized **0.41**. The weakest single layer per setting (−0.14 to 0.84) is disclosed. | fixed (recomputed) |
| 6 | W-E2 | New `sinktda/late_fusion.py`, a stacked test on out-of-fold logits, is shown as a new Table 3 column and in the prose. **Result:** 0D stacked onto NT changes AUC by 0.000 to +0.002 (equivalent in 11/11 settings; no 95% interval above 0). Deflated PH changes it by 0.000 to +0.001 (11/11). The abstract claim ("adds at most 0.003 AUC … whether fused early or late") is supported. | fixed (recomputed) |
| 7 | W-E4 + W-E5 | The prompt boundary is fixed (prefix tokenized without its trailing space; answer ≥ 1 token). bf16 is the default. TQA Qwen-3B, Phi-3, TinyLlama, SmolLM and both HaluEval settings were **re-extracted in bf16** (per-head features now also exist for HE Qwen-3B) and re-evaluated; the layer split and the worked example were rerun. All 11 settings now use bf16. The HaluEval rows with an empty answer are gone; the 195 remaining one-token rows are genuine one-word answers. | fixed (re-extracted) |
| 8 | New finding | New `sinktda/dtype_sensitivity.py` and appendix Table 4: fp16→bf16 changes topological AUCs by up to **0.058** (1D, TQA Qwen-3B; ≤ 0.005 elsewhere), sink features by ≤ 0.003, and non-topological banks by ≤ 0.001. | added |
| 9 | W-T2 (simulation) | The i.i.d. simulation now uses true Uniform[0,1] upper-triangular weights (20/8 draws per N); the new slope is **1.31**, versus 1.61 from the mis-specified triangular simulation. A random-causal-attention (no-sink) reference is added (slope **1.27**). The appendix Morse numbers are generated as a macro. | fixed (recomputed) |
| 10 | W-E9 | The scaling paragraph is data-driven: 10/11 settings grow significantly more slowly than i.i.d.; HE Qwen-3B does not; the exponent is > 1 on both HaluEval settings. The abstract no longer claims "sub-linear growth". | fixed |
| 11 | W-E3 | The seed-SD statement is now a macro (max 0.008; 17 cells exceed 0.004). The fixed-predictor bootstrap limitation, the C-threshold confound and the stacking optimism are stated. | fixed (text) |
| 12 | W-E6/E7/E8/E10/E11/E12 | These are now stated or qualified in the text: string-match labels are unaudited; the TinyLlama BOS mismatch is disclosed; SmolLM is called a "contrast", not a control; the b=0 planted-cycle failure is reported; the per-head test is relabelled P₀,h. "Four of five", "three of four", "any setting" and the 1D exceptions (now TQA Qw3B, HE Qw3B, HE Qw1.5B) are macro counts with significance. The Qwen system turn and HaluEval's lack of a chat template are described correctly. | fixed (text) |
| 13 | W-T3/T4/T5/T7 | An example attaining the bound is given ("attained"; "at most δ₀"). The rooted-tree MST argument handles ties. The distance remark requires a strictly increasing transform. The max-death bound is now checked (0 violations in 568,028 graphs), and both tolerances are stated. | fixed |
| 14 | W-P1–P4 | Times font via T1 fontenc; hyperref loaded last; predictions renamed C1–C4; TOST expanded; the margin is written m; the Fig. 2 legend says "Prop. 1"; the Fig. 1 legend is enlarged; Table 2's rounding bug (1.000 printed as ".000") is fixed; the stale "no per-head for HE" caption is removed; the title no longer has a manual break. | fixed |
| 15 | W-P5/P6 | README rewritten for `sinktda` (the v1 README is in `archive/README_v1.md`). Seeds (42–44 CV, 0 bootstrap, 0 TriviaQA sample), library versions, dtype and the run order (`sinktda/run_fix.sh`) are documented. | fixed |

Numbers quoted in Sections 1–7 above refer to the **pre-fix** fp16 extractions. After re-extraction, some values moved slightly:
- Sink−0D is equivalent in 5/10 settings (was 4).
- 1D|0D ranges from −0.008 to +0.033 and is equivalent in 7/10 (was 8).
- Deflated PH over Sink on TruthfulQA ranges from +0.029 to +0.045.

None of these changes reverses a conclusion. The fp16 outputs are kept in `archive/fp16_backup/`.

### What was not fixed, and why

- **W-E13 (scale ≥ 7B)** `[external]`: Mistral-7B does not fit in 16 GB. Run the commands in `sinktda/RUN_ON_5080.md`, copy `sinktda_out/truthfulqa_mistral*/` back, then run `python -m sinktda.evaluate truthfulqa_mistral truthfulqa_mistral_chat && python -m sinktda.late_fusion && python -m sinktda.report && python -m sinktda.appendix_extra`. The Mistral appendix still says the result is pending, and the abstract says "1.1B to 3.8B".
- **W-E6 (label audit)** `[external]`: needs human or LLM-judge labels. `sinktda_out/triviaqa_*/generations.csv` holds `question`, `aliases`, `pred` and `correct`. Sample 200 rows per model, label them, and report the disagreement rate.
- **W-E7 (BOS mismatch)**: disclosed rather than fixed. Fixing it would require regenerating TinyLlama's answers.
- **Training-variance-aware intervals** (per-seed OOF storage): nice-to-have, not done; the limitation is stated in the text.
- **Other baselines** (TOHA/HalluZig original code, semantic entropy, a SinkProbe comparison, a second on-policy dataset, per-head deflation): nice-to-have; not attempted on this hardware in this session.
- **Anonymity** `[external]`: the public GitHub repository and the third-party PDFs in `refrences/`. Make sure the submission links only the anonymous.4open.science mirror, and that the mirror does not include `refrences/`, `archive/`, or logs containing local paths.
- **Style file** `[external]`: `iclr2026_conference.sty` is still used; swap it if the target year differs.
- **Citations** `[external]`: final venue check for barbero2025first, kostenok2023, perez2022topological and gardinazzi2024persistent, whose arXiv IDs are consistent but venue status is unverified. Burns, Marks and Orgad could be updated to their published venues.
- **Git**: nothing was committed. The working tree contains all changes, including regenerated results; new files include `sinktda/late_fusion.py`, `sinktda/dtype_sensitivity.py`, `sinktda/run_fix.sh`, `paper/sink_dtype.tex` and `sinktda_results/{late_fusion,dtype_sensitivity,synthetic_reference_alpha,worked_example_layers}.csv`.

### Post-fix scores

| | Before | After | Justification |
|---|---|---|---|
| Soundness | 2 | **3** | The characterization error, length confound, mis-specified simulation, early-fusion-only tests, HaluEval boundary bug and dtype heterogeneity are all fixed, and every claim now matches a generated number. Label noise and refitting variance remain unquantified. |
| Presentation | 3 | **3** | The style, font, notation and wording issues are fixed. The paper is dense, and Tables 2–3 are still small. |
| Contribution | 2 | **2** | The positioning is now honest: the theory combines classical results, and SinkProbe already links sinks to detection. The remaining value is the precise topological reduction, the audit, and the new precision-sensitivity finding. |
| Overall | 5 | **6** (marginally above the acceptance threshold) | A careful, reproducible negative result with correct theory. A ≥7B model and a label audit would be needed for an 8. |
| Confidence | 4 | **4** | |

---

## 9. Revitalization (2026-09-17, contribution push)

The plan and its outcomes are in `REVITALIZE_PLAN.md`. I re-read the revised manuscript as the harshest reviewer in the pool would. All numbers below come from `sinktda_results/` via `paper/sink_numbers.tex`.

### What changed

| # | Change | Evidence |
|---|---|---|
| R1 | **New headline: TOHA (ACL 2026) is a first-order detector.** Proposition 1: TOHA's per-head MTop-Div/\|R\| equals the mean over response tokens of 1 − (largest prompt attention) whenever a prompt-level coning defect δ_P = 0, and lies within δ_P of it otherwise. When the sink is the top prompt token, this is the response sink score. | `sinktda/toha.py`, `toha_{checks,auc,comp}.csv`. Over 12,336,672 real head graphs (11 settings) there are **0 violations**. 57–88% of graphs are exactly prompt-coned, and on those the identity holds to 6·10⁻⁸. The median per-head ρ(d, π̄) is 0.954–0.998. |
| R2 | **Topology adds nothing to TOHA's first-order counterpart.** | Stacking d onto π̄: 0.000 to +0.007 (equivalent in 11/11). Supervised all-heads π̄ vs d: −0.007 to +0.004 (equivalent in 11/11). Supervised response-sink vs d: equivalent in 9/11; the exceptions are SmolLM (no sink) and one wide interval. |
| R3 | **Honest negatives about R2.** TOHA's discrete head selection is noisy. On π̄ it shifts AUC by −0.012 to +0.018: 3 settings significantly better, 2 worse, and non-inferior at 0.015 in only 9/11. On the sink score it loses up to 0.049 in sink models, because the selected heads often have a non-sink top prompt token (the sink is top for every response token in only 19–79% of their examples). TOHA adds up to +0.013 over the non-topological probes, with the 95% interval above 0 in 3/11. All of this is stated in §5.4 and in the abstract. | `toha_comp.csv` |
| R4 | **Whole-barcode corollary (Cor. 1).** d_B(dgm_k(D), star diagram) ≤ δ_s for every k; the elementwise 0D death sandwich and the Betti-0 sandwich hold; every d_B- or W₁-stable vectorization (landscapes, persistence images) is a function of one column up to δ_s. | Proof in App. A; `sinktda/check_theory.py`: 0 violations on 300 random matrices, including H₂. |
| R5 | **The coning defect as a detector** (Direction 3). Head-averaged δ₀ is weak (AUC 0.50–0.65). Per-head δ₀ is strong (0.69–0.96) and adds +0.006 to +0.027 beyond per-head sink mass (significant in 11/11), but only 0.000 to +0.008 beyond the non-topological probes (equivalent in 11/11). | `sinktda/defect_probe.py`, `defect_probe.csv` |
| R6 | **Causal sink-bias intervention** (App. F). A constant b ∈ {−2, 0, 2, 4} is added to key-0 logits in every head (TQA; Qwen-1.5B, TinyLlama). Suppressing the sink lowers the prompt-coned share (58→37%, 74→63%), the sink-top share (45→21%, 63→21%) and ρ(d, sink score) (0.89→0.61, 0.97→0.41). Strengthening it raises ρ to 0.96 and 0.996. TinyLlama is monotone in b; Qwen peaks at b=2. TOHA's AUC barely moves, and the TOHA-vs-sink-score AUC gap is *not* monotone, which is reported as such. | `toha_*_causal.csv`. The first run was **invalid**: the custom attention received no additive mask, so the attention was non-causal and produced 140k sandwich violations. It was discarded and fixed (the explicit causal mask is bit-identical to eager at b=0), a causality guard now aborts on non-causal attention, and the rerun has 0 violations. |
| R7 | Scale scripts: `RUN_ON_5080.md` now covers Mistral-7B, Qwen2.5-7B and Llama-3.1-8B (bf16, CPU offload for the last two), and `report.py` already knows the new setting names. | not run (no GPU access) |
| R8 | Reframing: new title and abstract, new contribution list, TOHA/HalluZig scope stated precisely (HalluZig uses graph homology, which the theorem does not cover). Prop. 1 (i.i.d.) and §5.3 (1D) moved to the appendix; the forest figure moved to the appendix. The main text still ends on p. 9. | |
| R9 | Bib: the TOHA author list was **wrong** (it listed seven unrelated names) and has been corrected from the ACL Anthology page. Bubenik (2015) and Adams et al. (2017) were added after verification. | |

Deferred: fp32/int8 precision study (Direction 4), HalluZig reimplementation, ≥7B runs.

### Post-revitalization scores (harshest reviewer)

| | Before (§8) | After | Justification |
|---|---|---|---|
| Soundness | 3 | **3** | The new identity is proved, unit-tested against ripser, and has 0 violations on 12.3M real head graphs and all intervention runs. TOHA is evaluated with its own selection algorithm inside the shared CV. On-policy labels are still unaudited, the intervals still treat the predictor as fixed, and TOHA is not run on its own RAG benchmarks (RAGTruth, CoQA, SQuAD) or at 7B. |
| Presentation | 3 | **3** | The story is now sharper (detectors → first-order statistics). The paper is still very dense: Table 4 is at the edge of legibility, and much material lives in the appendix. |
| Contribution | 2 | **3 (borderline)** | New and checkable: the published ACL 2026 topological detector is proved to be a first-order attention statistic up to a computable defect. This is confirmed on real heads, and it explains away TOHA's own "attention to ⟨s⟩ is weak" comparison. A whole-barcode corollary covers every stable vectorization. *Case for 2:* Proposition 1 is a short application of Theorem 1 to a quotient graph; SinkProbe already remarked on sinks in TOHA's MST; models are ≤3.8B; TOHA's own benchmarks are not used; and HalluZig, the other recent topological detector, is not covered. |
| Overall | 6 | **6** (upper end; 8 within reach) | A reviewer who accepts the TOHA reduction as consequential will move to 8, but only with evidence at the scale and on the benchmarks where TOHA claims SOTA. |
| Confidence | 4 | **4** | I re-derived Proposition 1 and Corollary 1, reran the numerical checks, and traced every new macro to its CSV. |

**What would lock in Contribution 3 (or reach 4):**
1. TOHA reduction results at 7–8B (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B are scripted in `RUN_ON_5080.md`). These are the models in TOHA's own tables.
2. The same test on a RAG benchmark TOHA reports (RAGTruth or CoQA), ideally using the authors' released code (github.com/sb-ai-lab/TOHA) to rule out a reimplementation gap.
3. A theorem, or a clean negative, for HalluZig's graph-homology zigzag.
4. An audited on-policy label subset.

### 5080 results (2026-09-17, second session)

Runs on the author's RTX 5080 (16 GB, bf16 throughout, nothing quantized). Wall-clock:
Mistral-7B 5 runs in 21 min; Qwen2.5-7B 4 runs in 30 min; re-extractions 12 min.

| # | Finding | Evidence |
|---|---|---|
| S1 | **Proposition 1 holds at 7B.** 6,570,272 new head graphs from 4 settings (Mistral-7B, Qwen2.5-7B x TruthfulQA, on-policy TriviaQA), **0 sandwich violations**. Exactly prompt-coned on 57-81%, and on those `d` equals the pi-bar score to <=6e-8. Median per-head rho(d, pi-bar) 0.980-0.998. Combined with the existing 11 settings: **18,906,944 head graphs, 0 violations**. | `sinktda_out/*/toha.npz`, `sinktda.toha.checks` |
| S2 | **Theorem 1 holds at 7B.** 16 settings, 838,356 layer graphs, **0 bound violations** of any kind (H1 length, P0 sandwich, max-death). | `sinktda_results/theory_*.csv` |
| S3 | **The reduction tightens with scale on Mistral.** Mistral-7B under `[INST]` is the most sink-dominated model tested: sink mass 0.75 (previous max 0.69), 98.2% of graphs exactly coned (previous max 88%), 99.99% empty H1. 0D vs Sink differ by +0.00005 (equivalent); 1D AUC is 0.5006; 0D adds -0.0003 over the non-topological bank (equivalent). | `theory_truthfulqa_mistral.csv`, `comp_truthfulqa_mistral.csv` |
| S4 | **The Mistral appendix claim is verified.** The previously unverified "99.99% of 1D features are exactly zero" audit is confirmed at `frac_cells_h1_zero` = 0.999924, and delta_0 = 0 is now *directly* verified on 98.2% of graphs, which the old text explicitly said had not been checked. | `theory_truthfulqa_mistral.csv` |
| S5 | **No reimplementation gap.** The authors' own `transform_attention_scores_to_distances` and `transform_distances_to_mtopdiv` (github.com/sb-ai-lab/TOHA), transcribed verbatim and run with ripser, agree with our dense-Prim score on 21,696 head graphs: max abs difference 1.2e-8 (exactly 0 on Mistral), correlation 1.00000000. | `sinktda/toha_native.py`, `sinktda_results/toha_native.csv` |
| S6 | **Qwen2.5-7B's sink is not on the first token.** Mean attention to token 0 is 0.008 (Qwen2.5-1.5B: 0.53; 3B: 0.50, same template). In 30/30 real rows the argmax attended column is token 2, the newline after `system`, carrying 0.56. Consequently TOHA's top prompt token is token 0 in ~0.0004% of head graphs and rho(d, sink score) is **negative** (-0.11, -0.16), while rho(d, pi-bar) stays 0.98-0.99. **The reduction is unaffected; only the identification of "max prompt attention" with "sink attention" fails.** | direct forward passes; `toha_checks` on the new settings |
| S7 | **Why s=0 fails there, precisely.** With apex 0, median delta_0 = 0.81 and nothing cones. With apex 2 on the full graph, median delta_2 = 0.068 and 20% cone. On the subgraph with tokens 0 and 1 removed, median delta = 0.0000 and **80% are exactly coned**. Tokens preceding the sink cannot attend to it, so they structurally break the cone. Theorem 1 (arbitrary apex) is untouched; it is Corollary 1's instantiation at s=0 that is model-specific. | 60 head-averaged graphs, Qwen2.5-7B |
| S8 | **Extraction bug found and fixed (did not affect any published number).** `apply_chat_template` emits BOS for Mistral and Llama-3.1, and feature extraction then added a second one, splitting the sink across two tokens. On `triviaqa_mistral` this moved the prompt-coned share 0.75 -> 0.69, the sink-is-top-prompt-token share 0.68 -> 0.18, and rho(d, sink) 0.97 -> 0.81. Fixed by `sinktda.data.encode`, verified token-for-token identical on all 11 pre-existing settings (Qwen has no BOS, Phi-3 adds none, TinyLlama and `[INST]` add one). **Llama-3.1-8B uses `generic_chat`, so this would have corrupted the headline 8B result.** | `sinktda/data.py`, README "Prompt tokenization" |

| S9 | **On-policy labels audited (2 of 6 TriviaQA settings).** 200 answers per setting, seed 0, judged by Qwen2.5-7B-Instruct. Disagreement with the alias string match: **14.0%** (Mistral-7B) and 5.5% (Qwen2.5-7B, where the judge is the same model that wrote the answers, so read it as a floor). On Mistral the string match is mostly too *lax*: 20 of 28 disagreements are answers it accepts and the judge rejects. Re-scoring the existing out-of-fold predictions on those rows under judge labels moves every bank by at most 0.058 AUC and **leaves all four dominance claims intact** (NONTOPO >= 0D, NONTOPO >= DEFL, HIDDEN >= 0D, NONTOPO >= PH_0D) in both settings; only near-tied banks reorder. The conclusions do not depend on the labelling, but the 14% rate should be stated. | `sinktda/label_audit.py`, `label_audit{,_sensitivity}.csv` |

| S10 | **Absolute claims re-checked against the new data.** "On every coned graph $H_1$ is empty" (C2) holds exactly: all 13 settings with coned cells have `frac_h1zero_given_coned` = 1.0. "Zero bound violations" holds over all 16 settings. "None of 0D / deflated PH adds more than X to the non-topological probes" holds and tightens: max |delta| is 0.0020 over TruthfulQA+TriviaQA and only 0.0005 among the new 7B settings. **One paragraph must change**: the "model without a first-token sink as a contrast" passage in `sink_results.tex` treats SmolLM as the single no-sink model, but Qwen2.5-7B is now a second one that behaves *differently* -- SmolLM is coned at no apex at all (`coned_any_apex` 0.0000) whereas Qwen2.5-7B is coned at some apex on ~10% of graphs (0.0978, 0.1003). The SmolLM sentence stays true; the framing does not. | `theory_*.csv`, `comp_*.csv` |

| S11 | **Latent reporting bug fixed before it could corrupt the paper.** `numbers_toha` defined the "sink models" group as *every setting except `truthfulqa_smollm`*, hardcoded by name, while `write_numbers` used a measured `mean_sink_mass > 0.2` threshold. That was equivalent only while SmolLM was the sole model without a first-token sink. Qwen2.5-7B is a second one, so on regeneration it would have been counted as sink-dominated and silently wrecked `TohaArgZeroRange` (40--75% -> 0--83%), `TohaRhoSinkRange` (0.78--0.99 -> -0.16--0.99) and `TohaSinkLoss*`. Both now use a shared `SINK_MASS_MIN` threshold read from the theory tables, and `TohaNoSinkModels` names the excluded models. Verified behaviour-preserving: on the committed 11-setting data the refactor reproduces the published 57--88%, 40--75%, 0.78--0.99, 0.049, 0.107 exactly. | `sinktda/report.py` |

| S12 | **Figure~2 would have silently dropped Llama-3.1-8B.** `fig_layers` plotted `tq[:8]` against an 8-colour palette. There are already 8 TruthfulQA settings, so adding Llama-3.1-8B would have produced a 9th that never appeared in the figure and no warning. The palette is extended to 10 (indices 0--7 unchanged, so existing figures keep their colours), the cap is now `len(SERIES)`, and the function warns if it ever truncates. Note `truthfulqa_mistral_chat` is, after the BOS fix, near-identical to `truthfulqa_mistral` (coned 98.20% vs 98.21%, sink 0.7547 vs 0.7548) -- consider dropping it from the figure as a redundant line. | `sinktda/report.py` |

**Not done in this session, and why.**
1. **Llama-3.1-8B**: gated, and no Hugging Face token exists on the machine; `huggingface-cli login` cannot be run non-interactively. Nothing else blocks it -- the offload path is smoke-tested and the BOS fix is in place.
2. **Steps 2-3 in full** (`toha evaluate`, `defect_probe`, `report`, `appendix_extra`, paper compile): `sinktda_out/` for the 11 small-model settings does not exist on this machine. `toha evaluate` and `defect_probe` glob that directory and rewrite their CSVs wholesale, so running them here would have replaced 11-setting results with 4-setting ones. They were deliberately **not** run; the committed CSVs are untouched and still carry 11 settings.
3. **TOHA on its own benchmarks** (CoQA/RAGTruth): their pipeline requires a Comet API key (it uploads experiment data to a third-party service), their pre-generated dataset CSVs, and defaults to the gated Llama-3.1-8B. Rather than improvise a different protocol, only the implementation-equivalence check (S5) was done.

### Scores after the 5080 session (harshest reviewer)

| | After Sec. 9 | Now | Justification |
|---|---|---|---|
| Soundness | 3 | **3** | The identity now survives 18.9M head graphs with 0 violations, including two 7B models, and our TOHA score is shown numerically identical to the authors' released code -- which removes the most obvious "you reimplemented it wrong" objection. Against that: a real extraction bug (S8) was sitting in the pipeline and was only caught because a 7B model happened to expose it, which is not reassuring about the ones not yet caught; on-policy labels are still unaudited; intervals still treat the predictor as fixed; and no model above 7.6B has been run. |
| Presentation | 3 | **3** | S6/S7 give the paper a much cleaner story about what "sink" means, but the main text must now carry two 7B models, a moved sink, and a tokenization caveat. Table 4 gains rows and is already at the edge of legibility. Nothing here is fixed until the tables are regenerated and the page budget rechecked. |
| Contribution | 3 (borderline) | **3** | Firmer than borderline. The reduction is confirmed at the scale TOHA itself uses, against TOHA's own code, and S6/S7 convert the paper's weakest rhetorical move ("the topology is the sink") into a precise, testable claim about *which* vertex. *Case for 2 stands:* still no result on TOHA's own RAG benchmarks, still no >=8B model, HalluZig still uncovered. |
| Overall | 6 | **6** | Unchanged, and honestly so. The new evidence removes two specific reviewer objections (scale, reimplementation) but not the one that matters most for an 8: TOHA is still not beaten on its own turf. |
| Confidence | 4 | **4** | I re-derived nothing new, but I checked the identity against the authors' code, verified the BOS fix token-for-token on every pre-existing setting, and confirmed S6 with direct forward passes rather than trusting the pipeline. |

**To finish, in order.** (a) copy `sinktda_out/` for the 11 settings onto the 5080 box, or copy the 5 new setting directories back to the Mac; (b) `huggingface-cli login` and run the Llama-3.1-8B block in `RUN_ON_5080.md`; (c) run `evaluate`/`late_fusion`/`defect_probe`/`toha evaluate`/`report`/`appendix_extra` **where all 16+ settings live**; (d) rewrite the Mistral appendix from `\MistralConed`/`\MistralHOneZero`, add S6/S7 to Sec. 5.4 and the abstract, and recheck the 9-page limit.

### Author to-do (by hand)

1. **5080 runs** (`sinktda/RUN_ON_5080.md`): TOHA extraction first, then full extraction, for Mistral-7B, then Llama-3.1-8B and Qwen2.5-7B (bf16, `SINKTDA_OFFLOAD=1` for the last two; Llama is gated, so run `huggingface-cli login`). Copy the outputs back, then run `evaluate`, `late_fusion`, `defect_probe`, `toha evaluate`, `report`, `appendix_extra`. After that, update "1.1B–3.8B" in the abstract and limitations and the Mistral appendix text.
2. **Anonymous mirror** (anonymous.4open.science/r/ICLR-BE07):
   - include `sinktda/`, `sinktda_results/`, `sinktda_out/` (or a link to the features), `paper/` sources, `README.md` and `requirements.txt`;
   - **exclude** `refrences/` (third-party PDFs), `logs/` (local paths), `archive/`, `.tmp/`, `HANDOFF.md`, `PROMPT_REVITALIZE.md`, `REVIEW_AC.md` and `REVITALIZE_PLAN.md`;
   - grep the mirror for your name, email and `/Volumes/`.
3. **Style file:** confirm `iclr2026_conference.sty` is the correct one for this cycle.
4. **Citations:**
   - check the venue for barbero2025first, kostenok2023, perez2022topological and gardinazzi2024persistent;
   - optionally update Burns (ICLR 2023), Marks (COLM 2024) and Orgad (ICLR 2025);
   - the TOHA, Bubenik 2015 and Adams 2017 entries were verified on 2026-09-17.
5. **Label audit** (optional but valuable): 200 on-policy answers per model.
6. **Git:** nothing was committed. New files:
   - `sinktda/{toha.py, defect_probe.py, check_theory.py, run_toha.sh, run_toha_causal.sh}`;
   - `sinktda_results/{toha_*.csv, defect_probe.csv, check_theory.csv}`;
   - `sinktda_out/*/toha.npz`, `sinktda_out/truthfulqa_*_sb*/`;
   - `REVITALIZE_PLAN.md`.
