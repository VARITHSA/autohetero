"""The same image pipeline with and without AutoHetero, side by side.

    python demo/compare.py

Runs demo/pipeline.py's four steps on a small and a large batch: first as plain PyTorch on the CPU, then by hand
on the GPU (.to("cuda"), if there is one), then with autohetero.enable(). It checks that all three give the same
answer and prints the median time of each.
"""
import statistics
import time

import torch

import autohetero
from autohetero import ops

BATCHES = [1, 64]  # small: the CPU should win; large: the GPU should win
REPS = 3


def pipeline(images):
    x = ops.sobel(ops.blur(images))
    x = ops.resize(x, scale=224 / min(x.shape[-2:]))
    return ops.resnet18(x).argmax(1)


def timed(fn, images):
    """(median seconds over REPS, result), after one warm-up run (CUDA start-up, model loading)."""
    fn(images)
    times = []
    for _ in range(REPS):
        start = time.perf_counter()
        out = fn(images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return statistics.median(times), out


def by_hand_on_gpu(images):
    return pipeline(images.to("cuda")).cpu()


inputs = {b: torch.rand(b, 3, 256, 256, generator=torch.Generator().manual_seed(0)) for b in BATCHES}
results = {b: {} for b in BATCHES}

for b, images in inputs.items():  # before enable(): ops run where their input is
    results[b]["Without (CPU)"] = timed(pipeline, images)
    if torch.cuda.is_available():
        results[b]["Without (GPU by hand)"] = timed(by_hand_on_gpu, images)

autohetero.enable()  # <- the one line
for b, images in inputs.items():
    results[b]["With AutoHetero"] = timed(pipeline, images)

print(f"\n{'batch':>5}  {'variant':<22} {'time':>10}  same answer as CPU")
for b, variants in results.items():
    reference = variants["Without (CPU)"][1]
    for name, (seconds, out) in variants.items():
        print(f"{b:>5}  {name:<22} {seconds * 1000:>7.1f} ms  {torch.equal(out.cpu(), reference)}")
print()
autohetero.report()
