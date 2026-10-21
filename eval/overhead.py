"""Dispatch overhead per call (RQ3, DESIGN.md §1, §10) on this machine.

    python eval/overhead.py                     # -> eval/out/overhead.csv

Each case times AutoHetero's own work around a call that does almost nothing (a 16-element input), minus the
same call made directly: features, the decision, moving data that is already in place, and handing it back.
  @auto function            a user function after its first call, decided by its own fit (never cached)
  built-in op               autohetero.ops.sort after enable(), decided by the forests and then cached
  decision miss             one forest decision for a size seen for the first time (the decision alone)
The < 1 ms target is a hypothesis to report against.
"""
import argparse
import csv
import os
import statistics
import timeit
from pathlib import Path

import numpy as np
import torch

import autohetero
from autohetero import devices, executor, features, history, ops, predictor

ROOT = Path(__file__).resolve().parent.parent


def measure(number=20_000, repeats=7):
    """({case: seconds per call}, {"@auto function": mode chosen, "built-in op": mode chosen})."""
    chosen = []

    def nothing(t):
        if not chosen:
            chosen.append(executor.running_on.get())
        return t

    per_call = lambda fn: statistics.median(timeit.repeat(fn, number=number, repeat=repeats)) / number
    x, xn, dispatched = torch.zeros(16), np.zeros(16, dtype=np.float32), autohetero.auto(nothing)
    table, raw_sort = torch.zeros(16, 4), ops.RAW["sort"]
    autohetero.enable()  # built-in ops dispatch too (and the machine is calibrated, once)
    dispatched(x), dispatched(xn), autohetero.ops.sort(table)  # first calls decide
    builtin_mode = history.last()["mode"]
    quiet = os.environ.get("AUTOHETERO_QUIET")
    os.environ["AUTOHETERO_QUIET"] = "1"  # printing the decision log isn't dispatch overhead
    try:
        direct = per_call(lambda: nothing(x))
        cases = {
            "direct call, no AutoHetero (baseline)": direct,
            "@auto function": per_call(lambda: dispatched(x)) - direct,
            "@auto function, numpy input": per_call(lambda: dispatched(xn)) - direct,
        }
        with autohetero.force("cpu"):
            cases["@auto function inside force('cpu')"] = per_call(lambda: dispatched(x)) - direct
        cases["built-in op, cached decision"] = per_call(lambda: autohetero.ops.sort(table)) - per_call(lambda: raw_sort(table))
        predictor._cache.clear()
        new = [features.of("sort", (torch.zeros(2**k, 4),), (None, None)) for k in range(5, 21)]  # made outside the timing
        cases["decision miss (the forests run)"] = statistics.median(
            timeit.timeit(lambda f=f: predictor.decide("sort", f), number=1) for f in new)
    finally:
        if quiet is None:
            del os.environ["AUTOHETERO_QUIET"]
        else:
            os.environ["AUTOHETERO_QUIET"] = quiet
    return cases, {"@auto function": "cpu" if chosen[0] == devices.CPU else "gpu", "built-in op": builtin_mode}


def main(out_dir):
    cases, modes = measure()
    note = "" if devices.GPU is not None else " (no GPU: decisions skip the forests)"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "overhead.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", "microseconds_per_call"])
        writer.writerows([case, f"{s * 1e6:.2f}"] for case, s in cases.items())
    print(f"decisions for the tiny inputs: {', '.join(f'{k} on {m.upper()}' for k, m in modes.items())}{note}")
    for case, s in cases.items():
        print(f"  {case:40s} {s * 1e6:10.2f} us")
    return out_dir / "overhead.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "out")
    print("wrote", main(parser.parse_args().out))
