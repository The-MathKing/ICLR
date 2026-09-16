"""
device.py -- CUDA-first device/dtype resolution.

Every extraction script in this repository was written on Apple Silicon and
selects `"mps" if torch.backends.mps.is_available() else "cpu"`. On an NVIDIA
box that expression is False, so the scripts silently run on CPU -- correct
results, ~50-100x slower, with no warning. This module fixes that in one place.

Usage:
    from rigor.device import get_device, get_dtype, hf_cache_note
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=get_dtype(device), attn_implementation="eager"
    ).to(device)

NOTE on attn_implementation: every attention-graph script needs
`output_attentions=True`, which is incompatible with SDPA/FlashAttention. Keep
`attn_implementation="eager"` or the attention tensors come back as None. At the
sequence lengths in this paper (N ~ 27-43 tokens) eager costs essentially
nothing.

NOTE on Blackwell (RTX 50-series, sm_120): requires torch >= 2.7 built for
cu128. Older wheels fail with "no kernel image is available for execution on
the device". `assert_gpu_usable()` below turns that into a clear message.
"""
import os
import warnings


def get_device(prefer: str = "auto") -> str:
    """Return the best available torch device string. CUDA is preferred."""
    import torch

    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    warnings.warn(
        "No GPU backend found; falling back to CPU. Extraction will be very slow.",
        RuntimeWarning,
    )
    return "cpu"


def get_dtype(device: str):
    """bfloat16 on CUDA (numerically safer than fp16 for logits/log-softmax),
    float16 on MPS, float32 on CPU."""
    import torch

    if device == "cuda":
        # bf16 is supported on every CUDA GPU this repo targets (Ampere+).
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def empty_cache(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def assert_gpu_usable(device: str) -> None:
    """Fail loudly and early rather than mid-extraction."""
    import torch

    if device != "cuda":
        return
    try:
        torch.zeros(8, device="cuda").sum().item()
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "CUDA device present but unusable. On RTX 50-series (Blackwell, sm_120) "
            "this usually means the installed torch predates sm_120 support. "
            "Install torch >= 2.7 for cu128:\n"
            "  pip install --index-url https://download.pytorch.org/whl/cu128 "
            "torch torchvision torchaudio\n"
            f"Underlying error: {exc}"
        ) from exc


def report(device: str) -> None:
    import torch

    print(f"[device] using: {device}  dtype: {get_dtype(device)}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(
            f"[device] {props.name}  "
            f"{props.total_memory / 1024**3:.1f} GiB  "
            f"sm_{props.major}{props.minor}  torch {torch.__version__}"
        )


def hf_cache_note() -> None:
    """The original scripts hardcode HF_HOME=/Volumes/2TB/hf_cache (an external
    drive on the authors' Mac). Respect the environment instead."""
    if "HF_HOME" not in os.environ:
        print(
            "[hf] HF_HOME not set; using the default cache (~/.cache/huggingface). "
            "Set HF_HOME to relocate model downloads."
        )
