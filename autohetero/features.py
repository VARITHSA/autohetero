"""What the predictor knows about a call (DESIGN.md §5).

The same function describes live calls and, through meta-device inputs, the rows of the benchmark and
calibration datasets, so training and runtime can't drift apart. No dispatch code here (§13).
"""
import math

from . import devices
from .ops import inputs

OPS = inputs.OPS + ("user",)
COLUMNS = ("elements", "bytes", "batch", "bytes_per_element", "cpu_load", "gpu_load", *(f"op={op}" for op in OPS))


def of(op, args, load, batch_dim=0):
    """Features of one call: op name ("user" for @auto functions), the arrays anywhere in args, and load.

    load is (CPU %, GPU %) from monitor.utilisation(); None means unknown. The first array is the reference
    for batch size and dtype, as it is for the output type (devices.restore).
    """
    found = list(devices.arrays(args))
    counts = [math.prod(a.shape) for a in found]
    ref = found[0] if found else None
    return {
        "op": op,
        "elements": sum(counts),
        "bytes": sum(n * a.dtype.itemsize for n, a in zip(counts, found)),
        "batch": ref.shape[batch_dim] if ref is not None and -ref.ndim <= batch_dim < ref.ndim else 1,
        "dtype": None if ref is None else str(ref.dtype).removeprefix("torch."),
        "cpu_load": load[0],
        "gpu_load": load[1],
    }


def size_bucket(f):
    """log2 of the input elements: calls within a factor of 2 in size share a bucket."""
    return int(math.log2(max(f["elements"], 1)))


def vector(f):
    """f as numbers in COLUMNS order. Unknown load is NaN: scikit-learn's forests handle missing values."""
    nan = float("nan")
    return [
        f["elements"],
        f["bytes"],
        f["batch"],
        f["bytes"] / max(f["elements"], 1),
        nan if f["cpu_load"] is None else f["cpu_load"],
        nan if f["gpu_load"] is None else f["gpu_load"],
        *(float(f["op"] == op) for op in OPS),
    ]
