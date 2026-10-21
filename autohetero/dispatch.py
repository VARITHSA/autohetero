"""The @auto wrapper (DESIGN.md §5): features -> decide -> run -> return, with force(), nesting and CPU fallback."""
import contextlib
import contextvars
import functools
import time

import numpy as np
import torch

from . import calibrate, devices, executor, features, history, monitor, predictor
from .calibrate import log
from .ops import SPLITTABLE

MODES = ("cpu", "gpu", "hybrid")

_forced = contextvars.ContextVar("autohetero_forced", default=None)  # per thread / task, so force() is thread-safe
_enabled = False
_unhealthy = set()  # (function name, log2 size bucket) where the GPU failed this session
_said = set()  # one-off log lines already printed


def enable():
    """Level 1: also dispatch the built-in autohetero.ops. Starts the load monitor and calibrates up front."""
    global _enabled
    _enabled = True
    monitor.start()
    try:
        calibrate.calibration()
    except RuntimeError as error:
        _gpu_unusable(error)
    if devices.GPU is None:
        _no_gpu()


def auto(fn=None, *, splittable=False, batch_dim=0):
    """Level 2: run every call of fn on whichever of CPU, GPU or Hybrid is predicted fastest.

    Use as @auto or @auto(splittable=True, batch_dim=0). splittable promises that fn's output along batch_dim
    depends only on the matching slice of its inputs; only then is a call chunked or split across devices.
    Every array argument is then split along batch_dim, and outputs are joined along batch_dim.
    Inputs and outputs keep their type (numpy stays numpy) and device. Rules for fn:
      - don't modify inputs in place: on the CPU they share memory with the caller's arrays;
      - take any model as an argument: a model in a closure or global can't be moved;
      - side effects may happen twice: if the GPU fails, the call reruns on the CPU.
    """
    if fn is None:
        return lambda f: auto(f, splittable=splittable, batch_dim=batch_dim)
    name = f"{fn.__module__}.{fn.__qualname__}"

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return call(name, "user", fn, args, kwargs, splittable, batch_dim)

    return wrapper


@contextlib.contextmanager
def force(mode):
    """Level 3: dispatched calls in this block (this thread or task only) run on mode: "cpu", "gpu" or "hybrid".

    "cpu" and "gpu" runs are never chunked, so they are pure single-device baselines.
    """
    if mode not in MODES:
        raise ValueError(f"force() takes one of {MODES}, not {mode!r}")
    token = _forced.set(mode)
    try:
        yield
    finally:
        _forced.reset(token)


def builtin(name, fn, args, kwargs):
    """A call of built-in op name: dispatched after enable() or inside an @auto call, else run where its input is."""
    if not _enabled and executor.running_on.get() is None:
        return fn(*args, **kwargs)
    return call(name, name, fn, args, kwargs, name in SPLITTABLE, batch_dim=0)


def call(name, op, fn, args, kwargs, splittable, batch_dim):
    """Run fn(*args, **kwargs) where it is predicted fastest and return its result in the caller's type and device."""
    like = devices.find_array((args, kwargs))
    if like is None:  # nothing to move
        return fn(*args, **kwargs)
    outer = executor.running_on.get()
    if outer is not None:  # nested: the outermost call already chose, so data isn't shuttled mid-function
        return executor.run_on(fn, args, kwargs, outer, like)
    monitor.start()
    f = features.of(op, (args, kwargs), monitor.utilisation(), batch_dim)
    mode = forced = _forced.get()
    times = {}
    if forced in (None, "hybrid"):  # a forced Hybrid split still needs the predicted speeds
        try:
            mode, times, new = predictor.decide(name, f, splittable)
        except RuntimeError as error:  # the GPU failed while calibrating (e.g. its memory is full)
            _gpu_unusable(error)
            mode, times, new = "cpu", {}, False
        if forced is None and new:
            log(f"{_short(name)}: batch {f['batch']}, {mode.upper()}")
        mode = forced or mode
    split = splittable and f["batch"] > 1 and bool(times)
    if mode == "hybrid" and not split:
        mode = min(("cpu", "gpu"), key=times.get) if times else "gpu"
    bucket = (name, features.size_bucket(f))
    if devices.GPU is None or bucket in _unhealthy:
        if devices.GPU is None:
            _no_gpu()
        mode, split = "cpu", False

    def piece(start, stop):
        if op == "user" or not args:  # user functions: every array argument splits
            return _cut((args, kwargs), start, stop, batch_dim)
        return (_cut(args[0], start, stop, batch_dim), *args[1:]), kwargs  # built-ins: arg 0 only (matmul: a)

    def part(start, stop, device):
        return executor.run_on(fn, *piece(start, stop), device, like)

    def predict(n):  # a piece's own predicted times: a small piece runs less efficiently than the whole batch
        load = (f["cpu_load"], f["gpu_load"])
        return predictor.decide(name, features.of(op, piece(0, n), load, batch_dim), splittable, quiet=True)[1]

    entry = {"name": name, **f, "mode": mode, "forced": forced is not None, "predicted": times,
             "fallback": False, "slowdown": None, "chunks_before_switch": None}
    start = time.perf_counter()
    if mode == "cpu" and not split:
        out = executor.run_on(fn, args, kwargs, devices.CPU, like)
        pieces = [("cpu", f["elements"], time.perf_counter() - start)]
    else:
        try:
            if split:
                out, info = executor.run(_short(name), part, predict, f["batch"], mode, batch_dim)
                pieces = [(d, f["elements"] * n / f["batch"], s) for d, n, s in info["pieces"]]
                entry["slowdown"], entry["chunks_before_switch"] = info["slowdown"], info["chunks_before_switch"]
            else:
                out = executor.run_on(fn, args, kwargs, devices.GPU, like)
                calibrate.sync(devices.GPU)  # the result may stay on the GPU: wait for it before reading the clock
                pieces = [("gpu", f["elements"], time.perf_counter() - start)]
        except RuntimeError as error:  # CUDA errors, OOM and driver errors are all RuntimeErrors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            rerun = time.perf_counter()
            out = executor.run_on(fn, args, kwargs, devices.CPU, like)  # if this raises too, the error is fn's own
            pieces = [("cpu", f["elements"], time.perf_counter() - rerun)]
            entry["fallback"] = True
            _unhealthy.add(bucket)
            reason = f"{type(error).__name__}: {str(error).splitlines()[0]}"
            log(f"{_short(name)}: GPU failed ({reason}); this size runs on the CPU from now on")
    history.record({**entry, "seconds": time.perf_counter() - start, "pieces": pieces})
    return out


def _gpu_unusable(error):
    """The GPU failed outside a call: stop using it this session, as on a machine without one. Never crash the caller."""
    devices.GPU = None
    _say_once(f"GPU unusable ({type(error).__name__}: {str(error).splitlines()[0]}); everything runs on the CPU this session")


def _no_gpu():
    """Say once that everything runs on the CPU: a CPU-only PyTorch on a GPU machine must not go unnoticed."""
    hint = "" if torch.cuda.is_available() else " (PyTorch sees no CUDA GPU: on a GPU machine, install a CUDA build from pytorch.org)"
    _say_once(f"no usable CUDA GPU: everything runs on the CPU{hint}")


def _cut(obj, start, stop, dim):
    """obj with every array in it cut to items [start, stop) along dim."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj[(slice(None),) * (dim % obj.ndim) + (slice(start, stop),)] if obj.ndim else obj
    return devices._map(obj, lambda x: _cut(x, start, stop, dim))


def _short(name):
    return name.rsplit(".", 1)[-1]


def _say_once(message):
    if message not in _said:
        _said.add(message)
        log(message)
