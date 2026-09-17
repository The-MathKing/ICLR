# Mistral-7B on the RTX 5080 box

Mistral-7B-Instruct-v0.2 in bf16 needs ~14.5 GiB, which does not fit on the 16 GB Mac.
Everything else in the study runs on the Mac.

```powershell
# torch >= 2.7 built for cu128 is required for Blackwell (sm_120)
pip install --index-url https://download.pytorch.org/whl/cu128 torch
pip install transformers datasets ripser scikit-learn pandas pyarrow scipy joblib
git pull
# native [INST] template (the published protocol) and the model's own chat template
python -m sinktda.extract --bench truthfulqa --model mistral
python -m sinktda.extract --bench truthfulqa --model mistral --template generic_chat --tag chat
# optional: on-policy
python -m sinktda.extract --bench triviaqa --model mistral --n 2000
```

Copy `sinktda_out/truthfulqa_mistral*/` (and `triviaqa_mistral/`) back to the Mac, then:

```bash
python -m sinktda.evaluate truthfulqa_mistral truthfulqa_mistral_chat
python -m sinktda.report
```

The per-head arrays for Mistral (32 layers x 32 heads) are ~40 MB compressed; fine to copy.
