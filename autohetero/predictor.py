"""Predicted time per device, and the CPU / GPU / Hybrid decision with its cache (DESIGN.md §6).

No dispatch code here: this module lifts out as a library (§13).
"""
import ast
import csv
import math
import statistics
import threading
from collections import defaultdict

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from . import calibrate, devices, features, history
from .ops import inputs

HYBRID_MARGIN = 0.1  # Hybrid must beat the faster single device by 10%; tune on benchmark data (week 7)
LOAD_BUCKET = 25  # load % per decision-cache bucket

_model = None  # trained on first use from this machine's calibration
_scalars = {}  # calibration's Hybrid costs: thread_seconds, concat_bytes_per_s
_cache = {}  # (name, log2 elements, dtype, cpu bucket, gpu bucket, splittable) -> (mode, seconds per element)
_modes = {}  # same key -> last mode decided, so the log shows each new decision and each change of mode
_lock = threading.RLock()  # re-entrant: _ensure_model holds it while train() takes it


class Model:
    """One random forest per device, and a decision tree alongside for the RQ1 comparison.

    Both learn log(seconds per input element). Beyond the largest size measured on a device, a forest then
    extrapolates time linearly instead of flat; with flat extrapolation, the few large CPU sizes calibration
    measures (§6) would make huge inputs look cheap on the CPU.
    """

    def __init__(self, rows):
        self.forests, self.trees = {}, {}  # forests[d].estimators_ stays reachable for the M.Tech work (§13)
        for device in sorted({d for _, d, _ in rows}):
            x = [features.vector(f) for f, d, _ in rows if d == device]
            y = [math.log(s / max(f["elements"], 1)) for f, d, s in rows if d == device]
            self.forests[device] = RandomForestRegressor(random_state=0).fit(x, y)
            self.trees[device] = DecisionTreeRegressor(random_state=0).fit(x, y)

    def predict(self, f, kind="forest"):
        """Predicted seconds for call f on each device."""
        x = [features.vector(f)]
        models = self.forests if kind == "forest" else self.trees
        return {d: math.exp(m.predict(x)[0]) * max(f["elements"], 1) for d, m in models.items()}


def rows(path):
    """Training rows (features, device, median seconds) from a calibration or bench/ dataset CSV."""
    runs = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            key = r["op"], r["size"], r["device"], r.get("cpu_load") or None, r.get("gpu_load") or None
            runs[key].append(float(r["seconds"]))
    out = []
    for (op, size, device, cpu, gpu), times in runs.items():
        load = (cpu and float(cpu), gpu and float(gpu))
        args = inputs.make(op, ast.literal_eval(size), device="meta")
        out.append((features.of(op, args, load), device, statistics.median(times)))
    return out


def train(training_rows, scalars):
    """Decide with a model trained on training_rows; scalars holds calibration's Hybrid costs. Clears the cache."""
    global _model, _scalars
    model = Model(training_rows)  # fitted outside the lock: decisions keep using the old model meanwhile
    with _lock:
        _model, _scalars = model, scalars
        _cache.clear()


def retrain():
    """Retrain on this machine's calibration plus its measured built-in op calls. Never runs inside a call."""
    info = calibrate.calibration()
    if info is not None:
        train(rows(info["timings"]) + history.rows(), info)


def decide(name, f, splittable=False, quiet=False):
    """(mode, predicted seconds per mode, whether the mode is new for this kind of call) for call f of name.

    quiet: a prediction for a piece of a call (the executor's chunks), not a decision; it doesn't count as one.

    Modes: "cpu", "gpu", "hybrid" (splittable calls), or "trial" for a user function's first splittable call.
    Built-in ops are predicted by the forests, cached per (size, dtype, load) bucket: one forest prediction
    takes milliseconds, too slow for the < 1 ms dispatch target (§6). User functions are predicted by their own
    fits from history once they have run (§6): microseconds, so never cached and never stale. Before that, a
    splittable call is a "trial" and any other call follows the forests; once one device is measured, the next
    call measures the other.
    """
    if devices.GPU is None:
        return "cpu", {}, False
    key = (name, features.size_bucket(f), f["dtype"], _bucket(f["cpu_load"]), _bucket(f["gpu_load"]), splittable)
    mode, times = _fitted(name, f, splittable) if f["op"] == "user" else (None, None)
    if mode is None:
        hit = _cache.get(key)
        if hit is None:
            _ensure_model()
            predicted = _model.predict(f)
            mode = _choose(predicted, f, splittable)
            if f["op"] == "user" and splittable and f["batch"] > 1:
                mode = "trial"  # nothing measured yet: time a piece on each device first
            hit = _cache[key] = (mode, {m: t / max(f["elements"], 1) for m, t in predicted.items()})
        mode, times = hit[0], {m: t * max(f["elements"], 1) for m, t in hit[1].items()}
    new = not quiet and _modes.get(key) != mode
    if not quiet:
        _modes[key] = mode
    return mode, times, new


def _fitted(name, f, splittable):
    """(mode, times) from a user function's own fits, or (None, None) before it has run on either device."""
    fitted = {d: history.predict(name, d, f["elements"]) for d in ("cpu", "gpu")}
    known = [d for d, t in fitted.items() if t is not None]
    if not known:
        return None, None
    _ensure_model()  # the Hybrid split costs come with it
    if len(known) == 1:
        return ("gpu" if known == ["cpu"] else "cpu"), {}  # measure the other device once
    return _choose(fitted, f, splittable), fitted


def _choose(times, f, splittable):
    """Fastest mode; adds the predicted "hybrid" time to times when the call is splittable."""
    best = min(("cpu", "gpu"), key=times.get)
    if splittable and f["batch"] > 1:  # a single item can't be split across devices
        c, g = times["cpu"], times["gpu"]
        split_cost = _scalars["thread_seconds"] + f["bytes"] / _scalars["concat_bytes_per_s"]
        times["hybrid"] = c * g / (c + g) + split_cost  # both devices finish together (§6)
        if times["hybrid"] < times[best] * (1 - HYBRID_MARGIN):
            best = "hybrid"
    return best


def _ensure_model():
    with _lock:
        if _model is None:
            info = calibrate.calibration()
            train(rows(info["timings"]), info)


def _bucket(load):
    return -1 if load is None else int(load // LOAD_BUCKET)
