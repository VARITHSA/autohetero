"""Inputs for the built-in ops at benchmark sizes (DESIGN.md §10), shared by calibration, bench/ and tests."""
import torch

IMAGE_OPS = ("blur", "resize", "sobel")
MODEL_OPS = ("resnet18", "mobilenetv2")
TABLE_OPS = ("filter", "sort", "groupby")
OPS = IMAGE_OPS + MODEL_OPS + TABLE_OPS + ("matmul", "fft")

RESOLUTIONS = [(64, 64), (256, 256), (720, 1280), (1080, 1920), (2160, 3840)]  # small -> 4K
GROUPS = 1000  # distinct keys in groupby inputs


def sizes(op):
    """Size series for op, each from small to large. A size is an input shape, a row count or a matrix side.

    Callers stop a series early once a run gets too slow or too big; matmul and fft run out of memory first.
    """
    if op in IMAGE_OPS:
        return [[(b, 3, h, w) for b in (1, 4, 16, 64, 256)] for h, w in RESOLUTIONS]
    if op in MODEL_OPS:
        return [[(2**i, 3, 224, 224) for i in range(9)]]  # batch 1 .. 256
    if op in TABLE_OPS:
        return [[round(10 ** (3 + i / 2)) for i in range(9)]]  # 10^3 .. 10^7 rows
    if op == "matmul":
        return [[2**i for i in range(6, 17)]]  # 64 .. 65536
    if op == "fft":
        return [[(2**i, 16384) for i in range(17)]]  # 1 .. 65536 signals of 16384 samples
    raise ValueError(f"unknown op {op!r}")


def make(op, size, generator=None, device="cpu"):
    """Args for op at size: float32 uniform in [0, 1). device="meta" gives shapes without allocating memory."""
    def rand(*shape):
        return torch.rand(*shape, generator=generator, device=device)

    if op in ("filter", "sort"):
        return (rand(size, 4),)
    if op == "groupby":
        return (torch.randint(0, GROUPS, (size,), generator=generator, device=device), rand(size))
    if op == "matmul":
        return (rand(size, size), rand(size, size))
    return (rand(*size),)  # image, model and fft ops take one tensor of shape size
