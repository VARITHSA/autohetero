"""Every measured call (DESIGN.md §5, §8): a log on disk, per-function totals, and online time fits.

Each call is appended to <machine cache>/history.jsonl with its predicted times next to the measured one: the
M.Tech confidence work is calibrated on exactly that gap (§13). Memory stays bounded however long the program
runs: only per-function totals and fit sums are kept. The fits are saved to fits.json at exit and read back by
the next session, so a user function's trial runs once per machine. No dispatch code here (§13).
"""
import atexit
import json
import math
import threading
import time
from collections import defaultdict
from pathlib import Path

from . import calibrate

DECAY = 0.95  # weight older points keep at each new one, so a fit follows a machine whose load changes
SLOPE = (0.0, 2.0)  # a fit's time-vs-size exponent stays in this range: noise between close sizes can't explode it
MAX_RETRAIN_RECORDS = 50_000
MAX_LOG_BYTES = 50 * 2**20  # past this the log moves to history.old.jsonl: disk use stays under ~100 MB

_lock = threading.Lock()
_file = None
_written = 0  # bytes in the current log file
_last = None  # the most recent record
_totals = defaultdict(lambda: defaultdict(float))  # name -> counters for report()
_fits = {}  # (name, device) -> decayed least-squares sums [n, sx, sy, sxx, sxy] of log time vs log elements
_loaded = False  # whether this session has read the fits saved by earlier ones


def record(entry):
    """Log one dispatched call.

    entry holds name, the call's features, mode, forced, predicted (seconds per mode, {} when not predicted),
    seconds (measured, whole call), pieces ([device, elements, seconds] per piece run), fallback and slowdown.
    """
    global _written, _last
    line = json.dumps({"time": round(time.time(), 3), **entry})
    with _lock:
        _last = entry
        if _written > MAX_LOG_BYTES:
            _rotate()
        _open().write(line + "\n")
        _written += len(line) + 1
        t = _totals[entry["name"]]
        t["calls"] += 1
        t["seconds"] += entry["seconds"]
        t[f"mode_{entry['mode']}"] += 1
        t["fallbacks"] += entry["fallback"]
        t["switches"] += entry["slowdown"] is not None
        _load_fits()
        # a user function with no fit yet was predicted by forests that never saw it: a guess, not an estimate
        guessed = entry["op"] == "user" and not any((entry["name"], d) in _fits for d in ("cpu", "gpu"))
        estimated = "gpu" in entry["predicted"] and not guessed
        t["predicted_calls"] += estimated
        if estimated:  # estimated saving vs always-GPU, from predicted times (§9)
            t["saved_vs_gpu_estimate"] += entry["predicted"]["gpu"] - entry["seconds"]
        for device, elements, seconds in entry["pieces"]:
            s = _fits.setdefault((entry["name"], device), [0.0] * 5)
            x, y = math.log(max(elements, 1)), math.log(max(seconds, 1e-9))
            for i, v in enumerate((1.0, x, y, x * x, x * y)):
                s[i] = s[i] * DECAY + v


def predict(name, device, elements):
    """Seconds for a call of name on device with this many input elements, from its fit; None if never run there."""
    with _lock:
        _load_fits()
        s = _fits.get((name, device))
        if s is None:
            return None
        n, sx, sy, sxx, sxy = s
    spread = sxx - sx * sx / n
    slope = (sxy - sx * sy / n) / spread if spread > 1e-6 * n else 1.0  # one size seen so far: assume linear
    slope = min(max(slope, SLOPE[0]), SLOPE[1])
    return math.exp(sy / n + slope * (math.log(max(elements, 1)) - sx / n))


def last():
    """The most recent record this session, or None."""
    with _lock:
        return _last


def totals():
    """{function name: {calls, seconds, mode_<mode>, fallbacks, switches, predicted_calls, saved_vs_gpu_estimate}}."""
    with _lock:
        return {name: dict(t) for name, t in _totals.items()}


def rows():
    """Training rows (features, device, seconds) from the built-in op calls on disk, newest last."""
    flush()
    logs = [calibrate.folder() / name for name in ("history.old.jsonl", "history.jsonl")]  # oldest first
    lines = [line for path in logs if path.exists() for line in path.read_text().splitlines()]
    out = []
    for line in lines[-MAX_RETRAIN_RECORDS:]:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:  # a line cut short by a crash mid-write
            continue
        if r["op"] == "user":  # user functions are predicted by their fits, not the forests
            continue
        for device, elements, seconds in r["pieces"]:
            share = elements / max(r["elements"], 1)
            f = {k: r[k] for k in ("op", "dtype", "cpu_load", "gpu_load")}
            f.update(elements=elements, bytes=r["bytes"] * share, batch=max(1, round(r["batch"] * share)))
            out.append((f, device, seconds))
    return out


def flush():
    with _lock:
        if _file is not None:
            _file.flush()


def save_fits():
    """Write the fits for the next session (at exit). Nothing is written when there are none."""
    with _lock:
        data = [[name, device, sums] for (name, device), sums in _fits.items()]
    if not data:
        return
    # shortcut: written only at exit, so a hard crash loses the session's fits, and when two programs share a
    # machine the last to exit wins; save after every call if either matters
    path = calibrate.folder() / "fits.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(path)  # never leaves a half-written file


def reset():
    """Forget this session's totals and fits and close the log, as if a new session started (in tests).

    Fits saved by earlier sessions are read again on next use.
    """
    global _file, _last, _loaded
    with _lock:
        if _file is not None:
            _file.close()
        _file, _last, _loaded = None, None, False
        _totals.clear()
        _fits.clear()


def _load_fits():
    """Read the fits saved by earlier sessions, once per session. Call with _lock held."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    try:
        for name, device, sums in json.loads((calibrate.folder() / "fits.json").read_text()):
            _fits.setdefault((name, device), sums)
    except (OSError, ValueError, TypeError):  # none saved yet, or a file this version can't read: start empty
        pass


def _rotate():
    global _file
    path = Path(_file.name)
    _file.close()
    _file = None
    path.replace(path.with_name("history.old.jsonl"))


def _open():
    global _file, _written
    if _file is None:
        path = calibrate.folder() / "history.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        _written = path.stat().st_size if path.exists() else 0
        # shortcut: buffered, so a hard crash loses the last few KB of log, flush per call if that matters
        _file = open(path, "a")
    return _file


atexit.register(flush)
atexit.register(save_fits)

