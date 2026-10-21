"""Scientific and signal workloads.

Both split along dim 0 of their first argument only: matmul by rows of a (b stays whole), fft by signal.
"""
import torch


def matmul(a, b):
    """a @ b."""
    return a @ b


def fft(signals):
    """Complex FFT of each signal along the last dim (a single 1-D signal does not split)."""
    return torch.fft.fft(signals)
