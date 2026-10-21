"""Controlled experiments (DESIGN.md §10, week 7). Run on the fixed machine only.

    python bench/experiments.py rq2 [--conditions idle,gpu,cpu] [--reps 5]
    python bench/experiments.py hybrid-gain [--reps 5]
    python bench/experiments.py all                     # -> bench/results/<host>-<time>.csv and .json (machine)

rq2: every condition x workload x strategy. Conditions: idle, gpu (a parallel PyTorch GPU job), cpu (stress-ng),
battery (switch battery saver on by hand when asked). Strategies: always-CPU, always-GPU, forced Hybrid
(splittable workloads) and AutoHetero; the oracle is the best measured option per run, computed in eval/.
hybrid-gain: idle machine, GPU-only vs Hybrid across batch sizes for every splittable op.

Strategies alternate run by run, so drift can't favour one. Rep 0 is the warm-up: it is recorded (first-call
costs such as trials stay visible) and eval/ leaves it out. AutoHetero's own overhead is inside its numbers:
every run is timed from the caller's side, inputs starting and results ending on the host.
"""
import argparse
import contextlib
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path

import torch

import autohetero
from autohetero import calibrate, devices, history, ops
from autohetero.ops import inputs

try:
    from bench import stress
except ImportError:  # run as a script: bench/ itself is on the path
    import stress

COLUMNS = ["experiment", "condition", "workload", "size", "strategy", "rep", "seconds",
           "mode", "slowdown", "chunks_before_switch", "detect_seconds", "fallback", "gpu_share"]

# Two sizes per op, chosen so even the CPU finishes a run in seconds; plus one user function (it has trials).
WORKLOADS = [
    *((op, s) for op in inputs.IMAGE_OPS for s in [(16, 3, 720, 1280), (64, 3, 1080, 1920)]),
    *((op, s) for op in inputs.MODEL_OPS for s in [(32, 3, 224, 224), (128, 3, 224, 224)]),
    *((op, s) for op in inputs.TABLE_OPS for s in [10**6, 10**7]),
    ("matmul", 2048), ("matmul", 8192),
    ("fft", (1024, 16384)), ("fft", (4096, 16384)),
    ("enhance", (32, 3, 720, 1280)),
]


@autohetero.auto(splittable=True)
def enhance(images):
    """The user-function workload: blur, then edges (built-in ops nested in one @auto call)."""
    return ops.sobel(ops.blur(images))


def rq2(write, conditions, reps=5, workloads=WORKLOADS):
    for condition in conditions:
        with _condition(condition):
            for name, size in workloads:
                fn = enhance if name == "enhance" else getattr(ops, name)
                splittable = name == "enhance" or name in ops.SPLITTABLE
                strategies = ["cpu", "gpu", *(["hybrid"] if splittable else []), "autohetero"]
                _alternate(write, "rq2", condition, name, size, fn, strategies, reps)


def hybrid_gain(write, reps=5, max_seconds=10.0, sizes=None):
    """Idle: GPU-only vs Hybrid for each splittable op, from its smallest size up until a run gets too slow.

    sizes ({op: [size, ...]}) limits the run to those ops and sizes.
    """
    default = {op: inputs.sizes(op)[1 if op in inputs.IMAGE_OPS else 0] for op in sorted(ops.SPLITTABLE)}  # images: 256x256
    for name, series in (sizes or default).items():
        for size in series:
            if not calibrate.fits(name, size, devices.GPU):
                break
            slowest = _alternate(write, "hybrid_gain", "idle", name, size, getattr(ops, name), ["gpu", "hybrid"], reps)
            if slowest > max_seconds:
                break


def _alternate(write, experiment, condition, name, size, fn, strategies, reps):
    """Run every strategy reps + 1 times, alternating; write one row per run. Returns the slowest run."""
    op = "blur" if name == "enhance" else name
    args = inputs.make(op, size, torch.Generator().manual_seed(0))
    slowest = 0.0
    for rep in range(reps + 1):
        for strategy in strategies:
            force = contextlib.nullcontext() if strategy == "autohetero" else autohetero.force(strategy)
            with force:
                start = time.perf_counter()
                fn(*args)
                seconds = time.perf_counter() - start
            r = history.last()
            chunks = r["chunks_before_switch"]
            detect = sum(p[2] for p in r["pieces"][:chunks]) if chunks else None
            share = sum(e for d, e, _ in r["pieces"] if d == "gpu") / sum(e for _, e, _ in r["pieces"])  # work on the GPU
            write([experiment, condition, name, size, strategy, rep, seconds,
                   r["mode"], r["slowdown"], chunks, detect, r["fallback"], share])
            slowest = max(slowest, seconds)
    print(f"{experiment:11s} {condition:7s} {name:12s} {str(size):22s} slowest run {slowest:8.3f} s", flush=True)
    return slowest


@contextlib.contextmanager
def _condition(condition):
    if condition == "gpu":
        with stress.gpu(device=str(devices.GPU)):
            yield
    elif condition == "cpu":
        with stress.cpu() as tool:
            print(f"cpu stress: {tool}", flush=True)
            yield
    elif condition == "battery":
        input("Switch battery saver ON (laptop with an NVIDIA GPU), then press Enter: ")
        yield
    elif condition == "idle":
        yield
    else:
        raise ValueError(f"unknown condition {condition!r}")


def main(which, conditions, reps, out_dir):
    if devices.GPU is None:
        raise SystemExit("the experiments compare CPU and GPU: run them on the machine with the NVIDIA GPU")
    autohetero.enable()  # calibrates once on a new machine
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{platform.node()}-{datetime.now():%Y%m%d-%H%M%S}"
    Path(f"{base}.json").write_text(json.dumps({**calibrate.machine(), "reps": reps, "conditions": conditions}, indent=2))
    with open(f"{base}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)

        def write(row):
            writer.writerow(row)
            f.flush()  # a long run that dies keeps everything measured so far

        if which in ("rq2", "all"):
            rq2(write, conditions, reps)
        if which in ("hybrid-gain", "all"):
            hybrid_gain(write, reps)
    return Path(f"{base}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("which", choices=("rq2", "hybrid-gain", "all"))
    parser.add_argument("--conditions", default="idle,gpu,cpu", help="rq2 conditions, comma-separated")
    parser.add_argument("--reps", type=int, default=5, help="timed runs per strategy, after one warm-up")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    a = parser.parse_args()
    print("wrote", main(a.which, a.conditions.split(","), a.reps, a.out))
