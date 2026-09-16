# `rigor/` — reviewer-hardening work for the MST-TDA audit

Everything here targets a specific weakness identified in review. **None of it has
been executed** — it was written in an environment without `torch`, `numpy`,
`sklearn` or `ripser`. Treat every script as a reviewed draft, not a verified
result. The one component whose logic *was* tested is
`stats_upgrades.permute_within_groups`, validated against a pure-Python mirror
over 4,000 randomized group layouts.

---

## 0. Prerequisite: get off CPU

`rigor/patch_device.py` ports the repo to CUDA. This is not optional on an
NVIDIA box — **15 scripts select `"mps" if torch.backends.mps.is_available()
else "cpu"` and none contains a CUDA code path**, so on an RTX 5080 every
extraction silently runs on CPU. 13 also hardcode
`HF_HOME = "/Volumes/2TB/hf_cache"`, an external drive on the authors' Mac.

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python rigor/patch_device.py --dry-run
python rigor/patch_device.py
```

RTX 50-series is Blackwell (sm_120) and needs **torch ≥ 2.7 / cu128**. Older
wheels fail with `no kernel image is available for execution on the device`;
`rigor.device.assert_gpu_usable()` converts that into a readable message.

Keep `attn_implementation="eager"` everywhere. SDPA/FlashAttention return
`attentions=None`, which would silently produce all-zero feature banks. At
N ≈ 27–43 tokens eager costs nothing.

---

## 1. Free wins — no GPU, run these first

| Script | Fixes |
|---|---|
| `logprob_baseline_full.py` | Roadmap item 1. Extends the paper's strongest baseline (AUC 0.984) from N=329 on one model to **all 7 settings at full N**, row-aligned to each setting's existing feature CSV on `(example_id, label)`. |
| `stats_upgrades.within_group_permutation_null` | Roadmap item 2. Replaces `master_pipeline`'s global `rng.permutation(y)` with a within-group permutation. The global shuffle breaks the one-positive-per-group design in ~34% of draws even in the minimal 2-group case, so Table 18's null is not calibrated to the design. |
| `stats_upgrades.bca_tost` | Roadmap item 3. Bias-corrected and accelerated bootstrap, with TOST read off the **90%** interval (the correct equivalence level at α=0.05). Removes the symmetry assumption that the published percentile-tail p-values depend on. |

Wiring `stats_upgrades` into `master_pipeline.py`:

```python
from rigor.stats_upgrades import bca_tost, within_group_permutation_null, skew_diagnostic
```

Report **both** the published percentile TOST and the BCa verdict. If they
agree — which `skew_diagnostic` will tell you — that agreement is itself a
robustness result worth one sentence. If they disagree anywhere, BCa is the
one to trust at 150 groups.

---

## 2. Hours on the 5080

### `rerun_mistral_native.py` — item 6, closes Limitation 1

The paper states it could not run fp16 Mistral-7B because 16GB of RAM cannot
hold a 14GB model. 32GB can. This re-extracts at native precision and prints the
1D degeneracy rate beside the int8 run.

```bash
python rigor/rerun_mistral_native.py          # GPU, bf16 (~14.5GiB weights, tight on 16GiB)
python rigor/rerun_mistral_native.py --cpu    # CPU, fp32 — the config the paper couldn't run
```

Either outcome is publishable. Still ~99.99% zero ⇒ the degeneracy is
architectural and the Mistral row stops being caveated. Materially lower ⇒ the
published "tightest equivalence in the paper" is a quantization artifact and
must be re-reported — better that you find that than a reviewer.

### Item 7 — harmonize N (highest rigor-per-GPU-hour)

Four `.head()` calls cap TruthfulQA below the available 817 questions:

| File | Line | Current | Change to |
|---|---|---|---|
| `phase10_qwen3b_eval.py` | 78 | `df.head(150)` | `df.head(817)` |
| `phase10_phi3_eval.py` | 80 | `df.head(150)` | `df.head(817)` |
| `phase10_tinyllama_eval.py` | 85 | `df.head(150)` | `df.head(817)` |
| `phase7_truthfulqa_smollm.py` | 72 | `df.head(500)` | `df.head(817)` |

Delete the corresponding cached CSVs to force re-extraction, then re-run
`master_pipeline.py`. Projected power at Δ=0.015, extrapolating the published
bootstrap SEs at 1/√n_groups (**projections, not measurements**):

| Setting | now | at 817 groups |
|---|---|---|
| TruthfulQA (Qwen2.5-3B) | 14.4% | ~55% |
| TruthfulQA (TinyLlama-1.1B) | 21.2% | ~77% |
| TruthfulQA (Phi-3-mini) | 42.7% | ~99% |
| TruthfulQA (SmolLM-1.7B) | 53.8% | ~75% |

This also retires the paper's own caveat that cross-model comparisons are
confounded with sample size. Note the honest limit: **Qwen2.5-3B still won't
reach 80%** — its Δ is intrinsically the noisiest, and it would need ~1,500
groups. TruthfulQA doesn't have them. That is the argument for an on-policy
benchmark (roadmap item 9), which has no N ceiling.

### `lookback_lens_baseline.py` — item 8

Lookback Lens is the most threatening missing baseline precisely because it is
attention-based and non-topological. If it ties or beats 0D-MST, the paper's
thesis gets *stronger*: attention graphs carry the signal, topology just isn't
how to extract it efficiently.

---

## Not written here

Roadmap items 9–12 (on-policy benchmark, EigenScore/INSIDE, semantic-entropy
probes, a second 7B model, an independent non-MST topological comparator) need
design decisions — grading protocol, sample count K, which model — that are
yours to make. Items 9, 10 and 11 should share a single sampled-generation pass;
generate once, compute all three.

---

## Before reporting any of this

Re-run the consistency check that caught the original defects: every number in
the manuscript should be traceable to a released CSV. The three that weren't
were `Δ<0.003` (false on 3 of 7 settings), the "all six model instances"
coverage claim, and Table 13's provenance.
