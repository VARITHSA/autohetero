"""First-run machine benchmark (DESIGN.md §6), cached per machine.

No dispatch code here: this module lifts out as a library (§13).
"""
import csv
import functools
import hashlib
import io
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pynvml
import torch

from . import devices, monitor, ops
from .ops import inputs

CACHE = Path.home() / ".cache" / "autohetero"
BUDGET = 30.0  # seconds for the op sweep (§6), split evenly across the ops
SIZES_PER_OP = 6
REPS = 3  # timed runs per size; fewer when an op's share of the budget runs out
MEMORY_FACTOR = 4  # inputs + device copy + output (a complex FFT output is twice its input)
COLUMNS = ["op", "size", "device", "elements", "cpu_load", "gpu_load", "rep", "seconds"]  # also the bench/ format

_lock = threading.Lock()


def calibration(force=False):
    """This machine's calibration: measured on first use (~30 s), then read from the cache. None without a GPU.

    Returns the machine, host<->GPU transfer rates, GPU launch latency, the costs Hybrid adds (a thread start
    and a concatenation), and "timings": the path of a CSV of op timings in the bench/ dataset format.
    """
    if devices.GPU is None:
        return None
    with _lock:
        info, here = machine(), folder()
        done = here / "calibration.json"
        if done.exists() and not force:
            return json.loads(done.read_text())
        log(f"Calibrating this machine once (~{BUDGET:.0f} s): {info['cpu']} + {info['gpu'] or devices.GPU}")
        start = time.perf_counter()
        here.mkdir(parents=True, exist_ok=True)
        result = {"machine": info, **_scalars(devices.GPU), "timings": str(here / "timings.csv")}
        _write(here / "timings.csv", _csv(_sweep_ops()))
        result["seconds"] = round(time.perf_counter() - start, 1)
        _write(done, json.dumps(result, indent=2))  # written last: its presence marks a finished calibration
        log(f"Calibration done in {result['seconds']} s, cached in {here}")
        return result


def folder():
    """This machine's cache folder, ~/.cache/autohetero/<machine key>: calibration and call history live here."""
    return CACHE / hashlib.sha1(json.dumps(machine(), sort_keys=True).encode()).hexdigest()[:12]


def machine():
    """What timings depend on (§6 machine key): CPU model, GPU name, NVIDIA driver, torch version."""
    gpu = devices.GPU is not None and devices.GPU.type == "cuda"
    return {
        "cpu": _cpu_name(),
        "gpu": torch.cuda.get_device_name(devices.GPU) if gpu else None,
        "driver": _nvidia_driver() if gpu else None,
        "torch": torch.__version__,
    }


def time_call(op, args, device, reps=5, warmup=1):
    """Seconds for each of reps runs of op(*args) on device, after warmup runs that are thrown away.

    Every run starts with args on the host and ends with the result back on the host, so transfers are
    included, exactly as for a dispatched call (§10).
    """
    like = devices.find_array(args)
    times = []
    for _ in range(warmup + reps):
        sync(device)
        start = time.perf_counter()
        devices.restore(op(*devices.to_device(args, device)), like=like)
        sync(device)
        times.append(time.perf_counter() - start)
    return times[warmup:]


def fits(op, size, device):
    """Whether op's inputs at size, their device copy and the output fit in free memory (host, and the GPU)."""
    need = MEMORY_FACTOR * sum(a.nelement() * a.element_size() for a in inputs.make(op, size, device="meta"))
    free = psutil.virtual_memory().available
    if device.type == "cuda":
        free = min(free, torch.cuda.mem_get_info(device)[0])
    return need <= free


def log(message):
    """The [AutoHetero] decision log (§2). AUTOHETERO_QUIET=1 silences it."""
    if os.environ.get("AUTOHETERO_QUIET") != "1":
        print(f"[AutoHetero] {message}", file=sys.stderr, flush=True)


def _scalars(gpu):
    x = torch.rand(16 * 2**20)  # 64 MB
    on_gpu, tiny, halves = x.to(gpu), torch.zeros(1, device=gpu), x.chunk(2)

    def thread():
        t = threading.Thread(target=int)
        t.start()
        t.join()

    return {
        "h2d_bytes_per_s": x.nbytes / _median_time(lambda: x.to(gpu), gpu),
        "d2h_bytes_per_s": x.nbytes / _median_time(lambda: on_gpu.to(devices.CPU), gpu),
        "launch_seconds": _median_time(lambda: tiny.add_(1), gpu, reps=100),
        "thread_seconds": _median_time(thread, devices.CPU, reps=20),
        "concat_bytes_per_s": x.nbytes / _median_time(lambda: torch.cat(halves), devices.CPU),
    }


def _median_time(fn, device, reps=5):
    fn()  # warm-up
    times = []
    for _ in range(reps):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return max(statistics.median(times), 1e-9)  # floor: with the CPU standing in for a GPU a copy is free


def _sweep_ops():
    monitor.start()
    targets, rows = {"cpu": devices.CPU, "gpu": devices.GPU}, []
    for name in inputs.OPS:
        start = time.perf_counter()
        counts = _sweep(name, targets, start + BUDGET / len(inputs.OPS), rows)
        sizes = ", ".join(f"{label} {n} sizes" for label, n in counts.items())
        log(f"  {name:12s} {sizes}  ({time.perf_counter() - start:.1f} s)")
    return rows


def _sweep(name, targets, deadline, rows):
    """Time op at up to SIZES_PER_OP sizes per device until deadline, appending rows. Returns sizes per device."""
    op, live, done, counts = ops.RAW[name], dict(targets), {}, dict.fromkeys(targets, 0)
    for size in _spread(name, targets.values()):
        elements = _elements(name, size)
        for label in list(live):
            if label in done and _predicted(done[label], elements) > deadline - time.perf_counter():
                del live[label]
        if not live:
            break
        args = inputs.make(name, size, torch.Generator().manual_seed(0))
        for label, device in list(live.items()):
            load = monitor.utilisation()
            try:
                if label not in done:
                    time_call(op, args, device, reps=0, warmup=1)  # first size: pay one-time costs untimed
                times = []
                for _ in range(REPS):
                    times += time_call(op, args, device, reps=1, warmup=0)
                    if time.perf_counter() + times[0] > deadline:
                        break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                del live[label]
                continue
            rows += [[name, size, label, elements, *load, rep, t] for rep, t in enumerate(times)]
            done.setdefault(label, []).append((elements, statistics.median(times)))
            counts[label] += 1
    return counts


def _predicted(done, elements):
    """One run's time at elements, scaled from a device's measured sizes by their growth rate (clamped to 1-2)."""
    (e1, t1), growth = done[-1], 1.5  # until two sizes show the real rate: matmul time grows as elements^1.5
    if len(done) > 1 and e1 > done[-2][0]:
        e0, t0 = done[-2]
        growth = min(2.0, max(1.0, math.log(t1 / t0) / math.log(e1 / e0)))
    return t1 * (elements / e1) ** growth


def _spread(op, on):
    """SIZES_PER_OP of op's benchmark sizes that fit in memory, evenly spread from smallest to largest input."""
    every = sorted(
        (s for series in inputs.sizes(op) for s in series if all(fits(op, s, d) for d in on)),
        key=lambda s: _elements(op, s),
    )
    picks = {round(i * (len(every) - 1) / (SIZES_PER_OP - 1)) for i in range(SIZES_PER_OP)}
    return [every[i] for i in sorted(picks)]


def _elements(op, size):
    return sum(a.nelement() for a in inputs.make(op, size, device="meta"))


def _csv(rows):
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    return out.getvalue()


def _write(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)  # atomic: readers never see a half-written file


@functools.cache  # forking sysctl from a process holding GBs of tensors can take ~0.4 s on macOS
def _cpu_name():
    if sys.platform == "darwin":
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    try:
        with open("/proc/cpuinfo") as f:
            return next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor()


def _nvidia_driver():
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlSystemGetDriverVersion()
    except pynvml.NVMLError:  # NVML missing or broken: the driver is recorded for information only
        return None


def sync(device):
    if device.type == "cuda":  # GPU work is asynchronous: wait for it before reading the clock
        torch.cuda.synchronize(device)
