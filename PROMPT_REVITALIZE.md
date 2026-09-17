# Prompt: raise the contribution of "The Topology Is the Sink" before the ICLR deadline

You are a senior ML researcher and co-author on an ICLR submission. Your job is to move this paper's **Contribution** score from 2 to at least 3 by the deadline, **2026-09-19**, which is about 2.5 working days away. You start with no context; everything you need is in `/Volumes/2TB/iclr`.

## Why the paper is at risk

A careful area-chair review (`REVIEW_AC.md`; read all of it, especially §4 "Novelty & framing" and §8 "Post-fix assessment") scored the paper:

| Soundness | Presentation | Contribution | Overall |
|---|---|---|---|
| 3 | 3 | **2** | 6 |

The science is now correct. The problem is that it reads as incremental:

- **Theorem 1 combines classical results.** Dominating-vertex cones (Barmak–Minian; Boissonnat–Pritam) plus one-sided stability.
- **Proposition 1's Θ(N) order is already known** (Hino–Kanazawa 2019). Only the 3/8 constant is new.
- **The directed-flag result is an easy observation.** The directed flag complex of a transitive tournament is the ordered simplex.
- **SinkProbe** (Binkowski et al., arXiv:2604.10697, 2026) already shows that attention sinks carry the signal of attention-based hallucination detectors.
- **The empirical story is a negative result on small models** (1.1B–3.8B): topology adds nothing beyond cheap probes.

A contribution-2 paper is usually rejected. We need at least one result that a reviewer would call new and consequential, not a better-written version of the same paper.

## What exists (read before proposing anything)

- **Manuscript:** `paper/paper.tex`, `paper/appendix_sink.tex`, `paper/sink_results.tex`. Number macros are generated into `paper/sink_numbers.tex` by `sinktda/report.py`, and appendix tables by `sinktda/appendix_extra.py`. `paper/paper_v1_audit.tex` is superseded.
- **Pipeline:** `sinktda/`. Read `README.md` first.
  - `extract.py` runs one forward pass per example and writes `sinktda_out/<setting>/`, which contains:
    - `layers.parquet`: per-layer PH, sink, δ₀, deflated and answer-only features;
    - `perhead.npz`: arrays of shape (rows, L, H) for MST total/max, sink mass, entropy, row max, **per-head δ₀**, LLM-Check and Lookback;
    - `hidden.npy`.
  - `evaluate.py` holds the grouped-CV, bootstrap and TOST machinery.
  - The other modules are `late_fusion.py`, `layer_split.py`, `synthetic.py`, `dtype_sensitivity.py` and `timing.py`.
- **Settings:** 11 settings, all in bf16. TruthfulQA ×5 models, HaluEval ×2, on-policy TriviaQA ×4.
- **Unused data you already have:**
  - Per-head δ₀ (`ph_delta0`) is extracted for every setting but never evaluated as a feature.
  - fp16 copies of five settings are in `archive/fp16_backup/`.
- **Papers:** `refrences/accepted/` and `refrences/denied/` hold third-party PDFs (for reading only; they must not ship with the submission).

## Hardware and rules (non-negotiable)

- **The Mac has 16 GB of unified memory.** Run one heavy job at a time. Use `--workers 4` and `SINKTDA_JOBS=2`, check `memory_pressure` first, and run jobs through the sequential-runner pattern in `sinktda/run_fix.sh`.
  - Env: `HF_HOME=/Volumes/2TB/hf_cache`, `TMPDIR=JOBLIB_TEMP_FOLDER=/Volumes/2TB/iclr/.tmp`.
  - Python: `.venv/bin/python -m ...`.
- **dtype is always bf16.** Models of 3.8B parameters or fewer fit on the Mac; fp32 fits only for models of 1.5B or fewer.
- **Models of 7B and up** (Mistral-7B, Llama-3.1-8B, Qwen2.5-7B) run only on the author's RTX 5080 (16 GB), which you cannot access. For those, write exact commands (following `sinktda/RUN_ON_5080.md`) and mark the results as pending. Design the paper so it stands without them, and gets stronger with them.
- **LaTeX:** compile with tectonic:
  `cd paper && TECTONIC_CACHE_DIR=/Volumes/2TB/iclr/.tmp/tectonic XDG_CACHE_HOME=/Volumes/2TB/iclr/.tmp tectonic -X compile paper.tex --outdir /Volumes/2TB/iclr/.tmp/build`
  Keep `\usepackage[T1]{fontenc}` before `times`. The main text must stay at 9 pages or fewer.
- **Honesty:**
  - Never invent a number, result or citation.
  - Every number in the text must come from a macro generated from a CSV.
  - If an idea fails, report it and drop it, or state it as a limitation. Do not spin it.
  - Verify any new citation on the web.
- **Git:** do not commit or push. Do not undo the fixes recorded in `REVIEW_AC.md` §8.

## Candidate directions (evaluate them; you may propose better ones)

Rank these by (expected contribution gain) × (probability it works) ÷ (hours needed), given 2.5 days:

1. **Hit the actual state of the art, not a strawman.**
   - Show that the reduction covers the methods reviewers care about: TOHA (ACL 2026; MTopDiv between prompt and response subgraphs) and HalluZig (EACL 2026; zigzag persistence across layers).
   - Prove what each method computes on coned graphs, e.g. whether MTopDiv becomes a function of the sink column restricted to the response tokens.
   - Then reimplement or run the method (use the authors' code if public) on our settings, and test whether its score is explained by sink statistics: rank correlation, and Sink vs. method under our TOST protocol.
   - If "SOTA topological detectors are sink detectors" holds for published ACL/EACL methods, the paper becomes consequential.
2. **Sharpen the theory into a full-diagram statement.**
   - On a coned graph, the entire VR barcode is determined by the sink column: 0D deaths are exactly {D_{0u}}, and there are no higher bars.
   - Therefore *every* vectorization (persistence images, landscapes, persistence entropy, Betti curves) is a function of sink attention.
   - Add a quantitative version for δ₀>0: a bottleneck/Wasserstein bound between the diagram and the "star diagram", in terms of δ₀.
   - This turns "PH ≈ sink" into a clean, general, citable statement. Check carefully that it is not already known.
3. **A positive, theory-derived signal.**
   - The coning defect itself (per-layer δ₀, per-head `ph_delta0`) and deflated per-head topology have never been evaluated as detectors.
   - Test them with the existing evaluator, both alone and under late fusion beyond NONTOPO/HIDDEN.
   - If δ₀ adds information that sinks and probes miss, the paper gains a constructive result. If not, report the null in one sentence.
4. **The numerical-fragility result.**
   - fp16→bf16 moved 1D AUC by up to 0.058, while non-topological features moved by ≤0.001.
   - Explain why: short H₁ bars live at the scale of δ₀ and are perturbation-sensitive; derive a stability-based bound.
   - Extend the experiment to fp32 (models of 1.5B or fewer) and int8/4-bit quantization if feasible.
   - Framed as "published TDA detector numbers are not reproducible across precisions, and here is why", this is a practical contribution.
5. **Causal evidence instead of correlational.**
   - Intervene on the sink inside a model, for example:
     - drop or replace the BOS token;
     - add a learned or constant bias to token-0 logits at inference;
     - use a register/"no-op" key.
   - Measure whether the TDA features and the TDA detector's AUC move exactly as the theorem predicts, while non-topological probes do not.
6. **Scale.** Prepare a turnkey 5080 script for Llama-3.1-8B-Instruct, Mistral-7B-Instruct and Qwen2.5-7B-Instruct on TruthfulQA and on-policy TriviaQA. Wire `report.py` so that the new settings appear automatically.

Also consider the framing. A title and abstract built around the strongest new result (e.g. "Topological hallucination detectors are sink detectors") may outrank a TDA-first framing. Decide this explicitly.

## How to work

1. **Orient (≤1.5 h).**
   - Read `REVIEW_AC.md`, the manuscript and `sinktda/`.
   - Skim the TOHA, HalluZig and SinkProbe papers; fetch them from arXiv if they are not in `refrences/`.
   - Run cheap checks that inform the choice. Direction 3 needs only the existing per-head arrays and a few minutes of CPU.
2. **Write a decision memo** to `REVITALIZE_PLAN.md` before any large change. It should contain:
   - the chosen directions (at most 3) and why;
   - what each would add to the contribution statement, in one sentence each;
   - the hours each needs, the risk that it fails, and what we report if it does;
   - the exact new experiments, with their commands and memory footprint;
   - which current material moves to the appendix to keep 9 pages;
   - the revised title, abstract and contribution list, as drafts.
3. **Execute in order of value**, with checkpoints.
   - After each result, update the macros in `write_numbers()`, regenerate with `python -m sinktda.report && python -m sinktda.appendix_extra`, and recompile.
   - Re-verify every qualitative word ("all", "every", "no") against the new tables.
4. **Finish.**
   - The main text must be at most 9 pages, with no overfull boxes and no undefined references; look at every main-text page as an image.
   - Update `README.md` and the reproducibility statement.
   - Append a "Revitalization" section to `REVIEW_AC.md` that re-scores the paper on the same ICLR scale (Soundness, Presentation, Contribution, Overall, Confidence), with one-line justifications. Be the harshest reviewer in the pool: if Contribution is still 2, say so and say what would fix it.
   - List everything the author must do by hand: 5080 runs, anonymizing the code mirror, removing `refrences/` and `logs/` from it, the style file, and final citation checks.

**Deliverables:**
- `REVITALIZE_PLAN.md`;
- the updated paper and code;
- the new section in `REVIEW_AC.md`;
- a short chat summary covering:
  - old vs. new scores;
  - the new headline contribution in one sentence;
  - which directions worked, failed or were deferred;
  - the author's to-do list.
