# GPU work for the LLM reviewer response (RTX 5080 box)

This is the list of review items that cannot be done on the 16 GB Mac. Everything else in
the review has already been fixed in the paper source and the analysis scripts; what is
left here needs either a 7B model resident in memory, long sequences, or repeated sampling.

Read `RUN_ON_5080.md` first — the environment, the offload flags, and the three gotchas at
the bottom of it still apply. This file only adds the new runs.

Items are ordered by how much they buy per hour of GPU. **A, B and C are runnable today
with the code in this repo.** D, E and F need a new benchmark loader or a new script; the
interface each one has to satisfy is spelled out so the rest of the pipeline picks the
results up without further changes.

Copy back the whole `sinktda_out/<setting>/` directory for every run, plus any
`sinktda_results/*.csv` the script writes. Nothing here should overwrite an existing
setting: every run below uses a new `--tag`, so the published numbers stay put.

---

## Summary

| # | Review item | Needs | Code exists? | Rough cost |
|---|---|---|---|---|
| A | Deflation deletes token 0 even when token 0 is not the sink | re-extraction with apex deflation | yes (`--apex-deflation`) | ~1.5 h |
| B | HaluEval is run without a chat template on instruction-tuned models | re-extraction with each model's template | yes (`--template`) | ~1 h |
| C | float16-vs-bfloat16 row for HaluEval Qwen2.5-3B mixes precision with a boundary change | one float16 re-extraction | yes (`--dtype float16`) | ~20 min |
| C2 | LLM-judge audit is only 200 rows per setting | full-validation-set judging | yes (`--n 2000`) | ~2 h |
| D | TOHA is evaluated off-distribution, never on long retrieved contexts | a RAG benchmark loader | **no** | ~1 day incl. code |
| E | No long-form / heavily context-grounded generation benchmark | a long-form loader + a claim-level labeller | **no** | ~2 days incl. code |
| F | No comparison against semantic entropy or CCS | a sampling script and a contrast-pair probe | **no** | ~2 days incl. code |

A, B, C and C2 close four reviewer points outright. D is the highest-value of the three
that need new code, because it tests the paper's central claim (Proposition 1) under the
conditions the detector was actually designed for.

---

## A. Apex deflation — "deflation hardcodes the deletion of token 0"

**What the reviewer said.** `DEFLATED PH` deletes token 0 to test prediction C4 ("removing
the sink exposes whatever non-sink topology exists"). For Qwen2.5-7B and SmolLM-1.7B token
0 is *not* the sink, so deleting it removes an ordinary token and does not test C4 for
those models.

**What is already in the paper.** The C4 numbers are now computed on sink-dominated
settings only, and the text says in both §5.2 and Appendix B that `DEFLATED` is a
first-token deletion rather than an apex deletion. That is honest but it is not the
experiment.

**What to run.** `sinktda/features.py` now has `deflate_vertex(A, k)` and
`extract.py --apex-deflation`, which deletes each layer's own best apex
`argmin_s delta_s` (already computed as `delta_argmin`) instead of token 0, and emits an
`apexdefl_*` block alongside the existing `defl_*` one. `evaluate.py` picks the block up
automatically as the bank `APEXDEFL` and runs three extra comparisons
(`T3b_apexdeflated_beyond_sink`, `T3c_apexdefl_vs_defl`, `I_apexdefl_beyond_nontopo`).
Settings without the block are unaffected.

The two models without a first-token sink are the point, but run a sink-dominated model
too, as a control: if apex deflation and token-0 deflation agree there, the difference
elsewhere is about the apex and not about the code path.

```powershell
$env:SINKTDA_OFFLOAD = "1"; $env:SINKTDA_GPU_MEM = "13GiB"
python -m sinktda.extract --bench truthfulqa --model qwen7b    --apex-deflation --tag apex --workers 4
python -m sinktda.extract --bench triviaqa   --model qwen7b    --apex-deflation --tag apex --n 2000 --workers 2

$env:SINKTDA_OFFLOAD = "0"
python -m sinktda.extract --bench truthfulqa --model smollm    --apex-deflation --tag apex --workers 4
python -m sinktda.extract --bench truthfulqa --model qwen3b    --apex-deflation --tag apex --workers 4   # control: token 0 IS the sink here
```

Cost: the same forward passes as before plus one extra ripser call per layer, so roughly
1.4x a normal extraction.

**Copy back.** `sinktda_out/{truthfulqa,triviaqa}_{qwen7b,smollm,qwen3b}_apex/`.

**Caveat worth knowing before you spend the time.** The apex is chosen per layer from the
*same* graph the features are computed on, so `APEXDEFL` is a label-free but
data-dependent choice of vertex. That is fine for the mechanistic claim (C4) and it is how
the reviewer framed it, but it is not a clean detector: report it as an intervention, not
as a bank that competes on AUC. The `T3c_apexdefl_vs_defl` comparison is the one to read.

---

## B. HaluEval with each model's chat template

**What the reviewer said.** HaluEval is run as a raw `Knowledge: ...\nQuestion: ...\nAnswer:`
string with no chat template. Every model evaluated on it is instruction-tuned, so the raw
format puts them outside the format they were tuned on, which could itself change the sink
behaviour the paper measures.

**Why it matters.** HaluEval is also the benchmark where the coning statistics are least
extreme, so if the raw format is what weakens the sink, that is worth knowing. (The paper
already draws no detection conclusion from HaluEval, because a TF-IDF model on the answer
text alone nearly saturates it.)

**What to run.** `data.halueval_rows` builds the prefix directly and ignores `template`,
so this needs a three-line change in `sinktda/data.py`: wrap the `Knowledge/Question`
block in the model's chat template through the same `_tqa_strings` helper the TruthfulQA
builder uses, gated on a new template value so the existing setting is untouched.

```python
def halueval_rows(tok, template, n=2000, chat=False):
    ...
    body = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}"
    if chat:
        full, pre = _tqa_strings(template, tok, body, a)      # per-model chat format
    else:
        pre = f"Knowledge: {row['knowledge']}\nQuestion: {row['question']}\nAnswer: "
        full = pre + a
```

Then add `--chat` to `extract.py` and pass it through. Run it for both HaluEval models:

```powershell
python -m sinktda.extract --bench halueval --model qwen3b   --n 1000 --chat --tag chat --workers 4
python -m sinktda.extract --bench halueval --model qwen1.5b --n 500  --chat --tag chat --workers 4
```

**What to report.** Mean sink attention, fraction of exactly coned layer graphs, and the
length-normalized `rho(P_0/(N-1), 1 - mbar)` for the raw and the chat format side by side.
The claim to test is "the format does not create the coning", so the right output is a
two-row comparison, not another AUC table.

**Run `check_tokenization.py` before anything else** — the prompt/response boundary is only
correct when the prefix and the full string tokenize the same way, and a chat template that
emits BOS has silently shifted it once already.

---

## C. A clean float16-vs-bfloat16 row for HaluEval Qwen2.5-3B

**What the reviewer said.** Table 5's HaluEval Qwen2.5-3B row mixes the precision effect
with a prompt–response boundary change, because the float16 extraction predates the
boundary fix. It therefore cannot be read as a precision measurement.

**Status.** The paper now says this explicitly, excludes that row from every maximum it
quotes, and notes that every number in it is at most 0.005 — so it drives none of the
headline numbers. The remaining fix is to make the row real.

**What to run.** One float16 extraction with the current (fixed) boundary, on CUDA.
float16 is not an option on MPS — Qwen produces non-finite values there and the extractor
refuses them — which is why this is a GPU item.

```powershell
mkdir archive\fp16_clean\sinktda_results
$env:SINKTDA_OUT = "archive\fp16_clean\sinktda_out"
$env:SINKTDA_RES = "archive\fp16_clean\sinktda_results"
python -m sinktda.extract  --bench halueval --model qwen3b --n 1000 --dtype float16 --workers 4
python -m sinktda.evaluate halueval_qwen3b
Remove-Item Env:SINKTDA_OUT, Env:SINKTDA_RES
```

**Do not let this land in the main `sinktda_results/`.** `report.load()` globs
`sinktda_results/auc_*.csv`, so a float16 setting placed there is swept into the main AUC
tables as if it were another model. `SINKTDA_RES` is new and exists for exactly this —
`extract.py` already honoured `SINKTDA_OUT`, but `evaluate.py` had its results directory
hardcoded, which is the trap. The layout mirrors the existing fp16 study
(`archive/fp16_backup/sinktda_results/`), which is where `dtype_sensitivity.py` reads from.

Once a clean row exists: point `dtype_sensitivity.OLD` at the new directory for that
setting, remove `halueval_qwen3b` from its `CONFOUNDED` list, re-run
`python -m sinktda.dtype_sensitivity`, and drop the two sentences in Appendix B that
explain why that row is excluded.

---

## C2. The LLM-judge audit on the full validation set

**What the reviewer said.** The audit of the on-policy TriviaQA labels judges 200
generations per setting, which puts the standard error of an AUC computed on the audited
rows near 0.03 — wide enough that near-ties between banks reorder. The judge is a model, so
scaling the audit is only a question of GPU time.

**What to run.** `label_audit.py` already takes `--n`; the judge is Qwen2.5-7B-Instruct,
which is what makes this a GPU item (it does not fit alongside anything else on 16 GB).

```powershell
$env:SINKTDA_OFFLOAD = "1"; $env:SINKTDA_GPU_MEM = "13GiB"
python -m sinktda.label_audit --n 2000
```

That judges every non-empty generation in every on-policy setting present in
`sinktda_out/`, so **make sure every on-policy setting's directory is on the box first** —
the script rebuilds `label_audit.csv` from whatever it finds, and a partial run silently
replaces the full table with a subset.

**Copy back.** `sinktda_results/label_audit.csv` and `label_audit_sensitivity.csv`.

**What changes in the paper.** Appendix F's disagreement rates get tight intervals, the
sentence about "the standard error of an AUC is near 0.03 with n = 200" goes away, and the
four dominance orderings the paper claims can be re-checked at full precision rather than
asserted with a caveat. Keep the caveat that the judge is a model and that one setting is
graded by the model that produced the answers.

---

## D. TOHA under its native conditions: long retrieved contexts

**This is the most valuable item on the list.** Proposition 1 says TOHA's divergence equals
one minus the response tokens' mean largest prompt attention, up to a prompt-level coning
defect `delta_P`. The paper verifies this on short-form QA, where it holds exactly on
57–88% of head graphs. TOHA was designed for retrieval-augmented generation. With a long
retrieved context the attention over the prompt can become diffuse, `delta_P` has more room
to grow, and the reduction could be materially looser — which is exactly the case the
reviewer, correctly, says is untested. If it still holds, the paper's central claim gets
much stronger. If it does not, that is a real boundary on the result and we should say so.

**What has to be written.** A row builder in `sinktda/data.py` that yields the same dict
shape as the others:

```python
{"example_id": int, "label": "grounded"|"hallucinated",
 "full": str, "prefix": str, "answer": str, "question": str}
```

with `prefix` being retrieved-context + question and `full` being prefix + response.
RAGTruth is the natural source (it has per-response hallucination labels over real
retrieved passages, with contexts in the 500–2,000 token range). Then add `"ragtruth"` to
the `--bench` choices in `extract.py` and `toha.py`.

**Memory, which is the real constraint here.** `perhead_features` needs the full
`(L, H, N, N)` attention tensor. In float32 that is `L*H*N^2*4` bytes: about 17 GB for a
32-layer, 32-head model at N = 2048. Two things follow.

1. `extract.py` now averages over heads on the device when `--no-perhead` is passed, so the
   per-layer path never materialises it (about 540 MB at N = 2048 instead of 17 GB). The
   `--max-len` flag replaces the hardcoded 1024-token skip.
2. The per-head path that TOHA needs has to stay per-head, but it is much cheaper than it
   looks: `toha.py` keeps only the *response* rows of the attention, so its buffer is
   `L*H*T*N` (T = response length), not `L*H*N^2`. At L = H = 32, T = 100, N = 2000 that
   is about 1.6 GB in float64 — fine. It is the response length, not the context length,
   that would blow it up.

**Both extractors skipped anything over 1,024 tokens.** That limit is now a `--max-len`
flag on both `extract.py` and `toha.py`, defaulting to the old 1,024 so nothing already
published moves. **Raise it for item D or every long example is silently dropped and you
get an empty-looking run**, which is precisely the failure mode that would make a RAG
benchmark look like it reproduced the short-form result.

Ripser is the other cost: a dense 2,000-point Vietoris–Rips filtration to `maxdim=1` is
slow and memory-hungry. For the TOHA check you do not need it — `d`, the pi-bar score,
`delta_P` and the prompt-coned fraction are all computed by dense Prim's and closed forms
in `toha.py`, with no ripser call. **So run item D through `toha.py` only**, and skip the
full `extract.py` feature bank at long N unless a detection number is also wanted.

```powershell
# after the loader exists
python -m sinktda.toha extract --bench ragtruth --model mistral --n 1000 --max-len 4096
python -m sinktda.toha extract --bench ragtruth --model qwen7b  --n 1000 --max-len 4096
```

**What to report, in order of importance.**

1. Violations of the Proposition 1 sandwich (expect zero; it is a theorem, and a non-zero
   count means an implementation bug, most likely in the prompt/response boundary).
2. The share of exactly prompt-coned head graphs (`delta_P = 0`) as a function of context
   length. Bucket by N and plot it. This is the number that answers the reviewer.
3. Median `delta_P`, and the largest `|d - pi-bar score|` on prompt-coned heads.
4. Median per-head rank correlation of `d` with the pi-bar score and with the response sink
   score.
5. Only then, AUC of `d` against the pi-bar score.

If the coned fraction falls off with context length, that is a publishable limitation and
should go into §6 with the curve, not be buried.

---

## E. A long-form generation benchmark

**What the reviewer said.** Everything is short-form QA (TruthfulQA, TriviaQA, HaluEval).
FActScore-style long-form generation or a heavily context-grounded task like XSum would
test whether the coning defect and the sink behaviour survive where attention is not
dominated by a single first-token sink.

**Why it is the hardest item.** It needs three things we do not have: long-form
generations, per-claim hallucination labels, and a way to turn claim-level labels into the
row-level binary label the pipeline expects. FActScore's own labelling uses a retrieval
pipeline plus a judge model.

**A cheaper version that answers most of the question.** The mechanistic claim does not
need hallucination labels at all. The theorem is about the attention graph. So:

- generate long-form outputs (biographies, or XSum-style summaries, 200–800 tokens) with
  the existing models;
- run the per-layer path only (`--no-perhead --max-len 2048`);
- report the fraction of exactly coned graphs, median `delta_0`, the share with empty
  `H_1`, and `rho(P_0/(N-1), 1 - mbar)` as a function of N.

That is a length-scaling study of Theorem 1 on real long generations, it needs no labels,
and it directly answers "does the sink still cone the filtration when the sequence is long".
The detection half (does topology beat cheap probes on FActScore) needs the labels and is a
separate, larger project — worth saying so rather than half-doing it.

Ripser at `maxdim=1` on N = 2048 is the bottleneck. If it is too slow, drop to `maxdim=0`:
`P_0`, `star_tot`, `delta_0` and the coned fraction are all 0D or closed-form, and `H_1`
being empty is already implied on coned graphs by Theorem 1(c).

---

## F. Semantic entropy and CCS baselines

**What the reviewer said.** The paper compares topological features against cheap
non-topological probes but not against standard uncertainty baselines: semantic entropy
(Farquhar et al., 2024) and CCS / latent knowledge probes (Burns et al., 2023).

**Scope note worth agreeing on before spending GPU time.** These are not attention-graph
methods, and the paper's claim is a reduction of topological features to first-order
attention statistics — adding them does not test that claim. The paper's limitations
section now says exactly this. They would contextualise the *detection* numbers in §5.3,
which is a weaker but legitimate reason. Treat this as optional, and do it last.

**Semantic entropy.** Per question: sample ~10 generations at temperature ~1.0, cluster
them by bidirectional entailment with an NLI model (`microsoft/deberta-large-mnli`), and
take the entropy over clusters. On the on-policy TriviaQA setup this is 10 extra
generations plus O(10^2) NLI calls per question, per model. At 2,000 questions that is the
dominant cost on this list; restrict it to two or three models. Output one scalar per row,
written as `sinktda_out/<setting>/semantic_entropy.npy` aligned to `layers.parquet` row
order, and add a `SEMENT` bank in `evaluate.banks()` plus a `B_sement` comparison against
`0D`.

**CCS.** Needs contrast pairs — the same question with "yes" and "no" completions — and the
hidden states of both. The stored `hidden.npy` is one formulation only, so this is new
forward passes, not a re-analysis. Two passes per row, mid-layer hidden states, then the
unsupervised consistency objective. Output the same way (`ccs.npy`, one scalar per row).

Both are label-free at fit time but must still be scored inside the same grouped
cross-validation as everything else, or the comparison is not like-for-like.

---

## Back on the Mac, after any of the above

```bash
# evaluate only the settings that were copied back; evaluate.py writes one file per setting
python -m sinktda.evaluate truthfulqa_qwen7b_apex triviaqa_qwen7b_apex truthfulqa_smollm_apex truthfulqa_qwen3b_apex
python -m sinktda.report && python -m sinktda.appendix_extra
```

`late_fusion.py`, `defect_probe.py` and `toha evaluate` rebuild their whole table from
whatever is in `sinktda_out/`, so run them only when every setting's directory is present.
`evaluate.py` and the new `c_grid.py` are safe to run on a subset.

New setting names need a line in `report.PRETTY` before they appear in any table.
