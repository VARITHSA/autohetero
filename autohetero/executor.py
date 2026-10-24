"""Running a call on a device, chunked execution, the slowdown check and Hybrid (DESIGN.md §7)."""
import contextvars
import math
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from . import calibrate, devices
from .calibrate import log

SLOWDOWN = 1.5  # a chunk this much slower than predicted moves the rest of the call to Hybrid
FIRST_CHUNK = 0.1  # share of the batch per single-device chunk...
MIN_CHUNK_SECONDS = 0.005  # ...but at least this long (Hybrid rounds too): shorter ones are dominated by launch overhead
HYBRID_ROUND = 0.25  # share of the batch per Hybrid round; speeds are re-measured and the split redone after each
# shortcut: all four are first guesses, to be tuned on benchmark data in week 7 (§7 asks for exactly that).

running_on = contextvars.ContextVar("autohetero_running_on", default=None)  # device of the outermost call
_gpu_feeder = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autohetero-gpu")


def run_on(fn, args, kwargs, device, like):
    """fn on device with its inputs moved there; the result comes back in like's type and device.

    Dispatched calls nested inside fn see running_on and stay on device.
    """
    token = running_on.set(device)
    try:
        out = fn(*devices.to_device(args, device), **devices.to_device(kwargs, device))
    finally:
        running_on.reset(token)
    return devices.restore(out, like)


def run(name, part, predict, batch, mode, dim):
    """A splittable call of batch items, run in pieces; part(start, stop, device) runs items [start, stop).

    predict(n) gives the predicted seconds of a piece of n items on each device. "trial" (a user function's
    first call): one piece of ~10% on the GPU, then one more on the CPU (GPU first: it works on a copy), then
    Hybrid by the two measured speeds. "cpu" / "gpu": chunks of ~10% of the batch (each >= 5 ms) on that device. From the second chunk on (the first also pays one-time costs
    such as copying a model), a chunk slower than SLOWDOWN x its own predicted time sends the rest of the call
    to Hybrid. "hybrid": rounds of ~25% (each >= 5 ms with both devices busy) split between both devices by
    speed, so they finish together; speeds are re-measured after every round. Pieces are joined along dim in their original order.

    Returns (output, info): info["pieces"] lists (device, items, seconds) in order; after a switch to Hybrid,
    info["slowdown"] is the ratio that caused it and info["chunks_before_switch"] how many chunks had run.
    """
    whole = predict(batch)
    both = whole["cpu"] * whole["gpu"] / max(whole["cpu"] + whole["gpu"], 1e-12)  # the whole batch split by speed (§6)
    # a round waits for its slower piece, so short rounds pay launch and sync costs several times over
    round_size = min(batch, max(2, math.ceil(HYBRID_ROUND * batch), math.ceil(MIN_CHUNK_SECONDS * batch / max(both, 1e-12))))
    speed = {d: round_size / max(t, 1e-12) for d, t in predict(round_size).items() if d in ("cpu", "gpu")}
    pieces, ran, done, switched, chunks = [], [], 0, None, None
    if mode == "trial":
        n = max(1, min(math.ceil(FIRST_CHUNK * batch), batch // 2))
        for device, start in (("gpu", 0), ("cpu", n)):
            out, seconds = _timed(part, start, start + n, device)
            pieces.append(out)
            ran.append((device, n, seconds))
            speed[device] = n / max(seconds, 1e-12)
        done = 2 * n
    elif mode != "hybrid":
        # whole sizes chunks only; each chunk is judged against its own prediction
        chunk = max(math.ceil(FIRST_CHUNK * batch), math.ceil(MIN_CHUNK_SECONDS * batch / max(whole[mode], 1e-12)))
        while done < batch:
            n = min(chunk, batch - done)
            out, seconds = _timed(part, done, done + n, mode)
            pieces.append(out)
            ran.append((mode, n, seconds))
            done += n
            slowdown = seconds / max(predict(n)[mode], 1e-12)
            if len(pieces) > 1 and slowdown > SLOWDOWN and done < batch:
                log(f"{name}: slowdown {slowdown:.1f}x, Hybrid")
                speed[mode], switched, chunks = n / seconds, slowdown, len(ran)
                break
    while done < batch:
        n = min(round_size, batch - done)
        g = round(n * speed["gpu"] / (speed["gpu"] + speed["cpu"]))  # gpu_share = s_gpu / (s_gpu + s_cpu), §6
        context = contextvars.copy_context()  # the GPU thread sees this call's force() and nesting state
        on_gpu = _gpu_feeder.submit(context.run, _timed, part, done, done + g, "gpu") if g else None
        on_cpu = _timed(part, done + g, done + n, "cpu") if g < n else None
        for result, device, count in ((on_gpu and on_gpu.result(), "gpu", g), (on_cpu, "cpu", n - g)):
            if count:
                pieces.append(result[0])
                ran.append((device, count, result[1]))
                speed[device] = count / max(result[1], 1e-12)
        done += n
    out = pieces[0] if len(pieces) == 1 else _join(pieces, dim)
    return out, {"pieces": ran, "slowdown": switched, "chunks_before_switch": chunks}


def _timed(part, start, stop, label):
    device = devices.CPU if label == "cpu" else devices.GPU
    begin = time.perf_counter()
    out = part(start, stop, device)
    calibrate.sync(device)  # the result may stay on the GPU: wait for it before reading the clock
    return out, time.perf_counter() - begin


def _join(pieces, dim):
    first = pieces[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(pieces, dim)
    if isinstance(first, np.ndarray):
        return np.concatenate(pieces, dim)
    if isinstance(first, dict):
        return {k: _join([p[k] for p in pieces], dim) for k in first}
    if isinstance(first, (list, tuple)):
        joined = [_join(list(items), dim) for items in zip(*pieces)]
        return type(first)(*joined) if hasattr(first, "_fields") else type(first)(joined)
    raise TypeError(f"a splittable function must return arrays (or tuples, lists or dicts of them), not {type(first).__name__}")
