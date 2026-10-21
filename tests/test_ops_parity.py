"""Phase 1-2 exit check (DESIGN.md §11): every op gives matching CPU and GPU results at 3 sizes.

Run on a machine with an NVIDIA GPU:  pytest -m gpu tests/test_ops_parity.py
"""
import pytest
import torch

from autohetero import devices, ops
from autohetero.ops import inputs

RTOL = 1e-4  # relative to the largest output value: MobileNetV2's random-weight logits are ~1e-9

IMAGES = [(1, 3, 64, 64), (8, 3, 256, 256), (2, 3, 1080, 1920)]
SIZES = {
    **{op: IMAGES for op in inputs.IMAGE_OPS},
    **{op: [(1, 3, 224, 224), (8, 3, 224, 224), (32, 3, 224, 224)] for op in inputs.MODEL_OPS},
    **{op: [10**3, 10**5, 10**6] for op in inputs.TABLE_OPS},
    "matmul": [64, 512, 2048],
    "fft": [(1, 1024), (64, 4096), (256, 65536)],
}


@pytest.fixture(autouse=True)
def no_tf32(monkeypatch):
    # TF32 (default for cuDNN convs on Ampere+) rounds to ~1e-3; this check is about the op code, not TF32.
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


@pytest.mark.gpu
@pytest.mark.parametrize("name, size", [pytest.param(op, s, id=f"{op}-{s}") for op in inputs.OPS for s in SIZES[op]])
def test_cpu_and_gpu_agree(name, size):
    op = ops.RAW[name]
    args = inputs.make(name, size, torch.Generator().manual_seed(0))
    on_cpu = op(*args)
    on_gpu = devices.restore(op(*devices.to_device(args, devices.GPU)), like=args[0])
    as_tuple = lambda out: out if isinstance(out, tuple) else (out,)  # groupby returns (keys, sums)
    for c, g in zip(as_tuple(on_cpu), as_tuple(on_gpu), strict=True):
        if c.is_floating_point() or c.is_complex():
            torch.testing.assert_close(g, c, rtol=RTOL, atol=RTOL * c.abs().max().item())
        else:
            assert torch.equal(g, c)
