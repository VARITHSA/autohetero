"""Collect the benchmark dataset (DESIGN.md §10): every built-in op x size x {CPU, GPU}, 5 timed runs each.

    python bench/collect.py                    # all ops -> bench/data/<host>-<time>.csv and .json (machine)
    python bench/collect.py --ops blur,sort

Final datasets come from the fixed machine only; Colab is for development.
"""
import argparse
import csv
import json
import platform
import statistics
from datetime import datetime
from pathlib import Path

import torch

from autohetero import calibrate, devices, monitor, ops
from autohetero.ops import inputs

MAX_RUN_SECONDS = 10.0  # a series stops growing on a device once one run there takes longer than this


def collect(op_names, out_dir, max_seconds=MAX_RUN_SECONDS):
    """Time every size of every op on CPU and GPU, appending rows as it goes. Returns the CSV path."""
    targets = {"cpu": devices.CPU} if devices.GPU is None else {"cpu": devices.CPU, "gpu": devices.GPU}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{platform.node()}-{datetime.now():%Y%m%d-%H%M%S}"
    info = {**calibrate.machine(), "devices": list(targets), "max_run_seconds": max_seconds}
    Path(f"{base}.json").write_text(json.dumps(info, indent=2))
    print(info, flush=True)
    monitor.start()
    with open(f"{base}.csv", "w", newline="") as f:
        rows = csv.writer(f)
        # cpu_load / gpu_load: monitor sample just before the timings (empty = unknown), the predictor's load features
        rows.writerow(calibrate.COLUMNS)
        for name in op_names:
            for series in inputs.sizes(name):
                live = dict(targets)
                for size in series:
                    if not live or not calibrate.fits(name, size, devices.CPU):
                        break
                    args = inputs.make(name, size, torch.Generator().manual_seed(0))
                    for label, device in list(live.items()):
                        load = monitor.utilisation()
                        times = _time(name, size, args, device, max_seconds)
                        if times is None:
                            del live[label]
                            continue
                        elements = sum(a.nelement() for a in args)
                        rows.writerows([name, size, label, elements, *load, rep, t] for rep, t in enumerate(times))
                        f.flush()  # a long run that dies keeps everything measured so far
                        print(f"{name:12s} {str(size):24s} {label}  {statistics.median(times) * 1e3:10.2f} ms", flush=True)
    return Path(f"{base}.csv")


def _time(name, size, args, device, max_seconds):
    """Run times on device, or None when this size is too big or too slow there (its series stops)."""
    if not calibrate.fits(name, size, device):
        return None
    op = ops.RAW[name]
    try:
        # Judge on a warm run: a cold first run also pays one-time costs (building a model, loading kernels).
        if calibrate.time_call(op, args, device, reps=1, warmup=1)[0] > max_seconds:
            return None
        return calibrate.time_call(op, args, device, warmup=0)  # already warm
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ops", default=",".join(inputs.OPS), help="comma-separated op names (default: all)")
    parser.add_argument("--out", default=Path(__file__).parent / "data", help="output folder")
    parser.add_argument("--max-seconds", type=float, default=MAX_RUN_SECONDS, help="slowest single run allowed")
    a = parser.parse_args()
    print("wrote", collect(a.ops.split(","), a.out, a.max_seconds))
