"""Background CPU / GPU load sampler (DESIGN.md §8). Calls read the cached sample: no syscalls on the hot path.

No dispatch code here: this module lifts out as a library (§13).
"""
import collections
import statistics
import threading
import time

import psutil
import pynvml
import torch

PERIOD = 0.2  # seconds between samples
WINDOW = 5  # report the median of the last ~1 s, so one odd sample can't flip a decision
# shortcut: on macOS psutil reads 0.0 for bursts of ~0.6 s, longer than the window covers; Linux (the fixed
# machine) reads /proc/stat and is unaffected. Revisit only if a macOS machine is ever used for final numbers.

_sample = (None, None)  # (CPU %, GPU %), None while unknown
_thread = None
_lock = threading.Lock()


def start():
    """Start the sampler (once per process). Returns after the first sample is in."""
    global _thread
    with _lock:
        if _thread is None:
            first = threading.Event()
            _thread = threading.Thread(target=_run, args=(first,), name="autohetero-monitor", daemon=True)
            _thread.start()
            first.wait(timeout=5 * PERIOD)


def utilisation():
    """Latest (CPU %, GPU %), each 0-100 or None if unknown (sampler not started, or no NVIDIA GPU).

    CPU is the load from *other* processes: this process's own work is subtracted, so a loop of
    CPU-dispatched calls does not look like a busy CPU to the next decision.
    """
    return _sample


def _run(first):
    global _sample
    me, cores = psutil.Process(), psutil.cpu_count() or 1
    gpu = _nvml_handle()
    recent = collections.deque(maxlen=WINDOW)
    psutil.cpu_percent(None), me.cpu_percent(None)  # the first readings only set the baseline
    while True:
        time.sleep(PERIOD)
        others = max(0.0, psutil.cpu_percent(None) - me.cpu_percent(None) / cores)
        recent.append((others, _gpu_load(gpu)))
        _sample = tuple(_median(column) for column in zip(*recent))
        first.set()


def _median(values):
    known = [v for v in values if v is not None]
    return statistics.median(known) if known else None


def _nvml_handle():
    if not torch.cuda.is_available():
        return None
    try:
        pynvml.nvmlInit()
        # shortcut: assumes NVML's device 0 is torch's cuda:0 (one GPU in scope, §3), match by UUID if
        # CUDA_VISIBLE_DEVICES is ever used to pick a GPU.
        return pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError:  # no NVML or driver: GPU load stays unknown
        return None


def _gpu_load(handle):
    if handle is None:
        return None
    try:
        # shortcut: whole-GPU load includes this process's own kernels, sample per process
        # (nvmlDeviceGetProcessUtilization) if that misleads the predictor.
        return float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
    except pynvml.NVMLError:
        return None
