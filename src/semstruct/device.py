"""Select CUDA, then Apple MPS, then CPU for the manual training loop.

The Hugging Face Trainer applies the same practical outcome: NVIDIA machines
have no MPS backend, and Apple Silicon uses MPS.
"""

from __future__ import annotations


def select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def release_device_cache() -> None:
    """Drop cached accelerator blocks after a dataset finishes.

    A single process that fine-tunes every database otherwise keeps the
    previous model's MPS or CUDA cache. On an 8 GB machine that cache
    forces the next run into swap.
    """
    import gc

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()
