"""Move a call's inputs to a device and its outputs back, converting numpy <-> torch.

Read ``devices.GPU`` at call time (not ``from autohetero.devices import GPU``): tests swap in the CPU as a
fake GPU so GPU code paths run on CPU-only machines.
"""
import copy
import itertools
import threading
import weakref

import numpy as np
import torch
from torch import nn

CPU = torch.device("cpu")
GPU = torch.device("cuda", 0) if torch.cuda.is_available() else None  # one GPU in scope (DESIGN.md §3)

_models = weakref.WeakKeyDictionary()  # user model -> {device: cached copy}
_models_lock = threading.Lock()


def to_device(obj, device):
    """obj with every tensor, ndarray and nn.Module inside it (lists, tuples, dicts) on device.

    ndarrays become tensors. The user's own models are never moved; a cached copy is used instead.
    On the CPU, tensors share memory with the caller's arrays, which is why @auto functions must not modify
    their inputs in place (DESIGN.md §2).
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, np.ndarray):
        try:  # copy: torch rejects negative strides (np.ascontiguousarray would turn 0-d arrays into 1-d)
            return torch.from_numpy(obj if obj.flags.c_contiguous else obj.copy()).to(device)
        except (TypeError, ValueError):  # no torch equivalent (object, str, datetime, longdouble, big-endian)
            return obj
    if isinstance(obj, nn.Module):
        return model_on(obj, device)
    return _map(obj, lambda x: to_device(x, device))


def restore(obj, like):
    """obj with every tensor inside it turned back into like's type and device.

    like is the call's reference input (see find_array): an ndarray gives numpy outputs on the CPU.
    """
    if isinstance(obj, torch.Tensor):
        return obj.numpy(force=True) if isinstance(like, np.ndarray) else obj.to(like.device)
    return _map(obj, lambda x: restore(x, like))


def arrays(obj):
    """Every tensor and ndarray in obj (searching lists, tuples, dicts), in order."""
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        yield obj
    elif isinstance(obj, (list, tuple, dict)):
        for x in obj.values() if isinstance(obj, dict) else obj:
            yield from arrays(x)


def find_array(obj):
    """The first tensor or ndarray in obj (searching lists, tuples, dicts), or None."""
    return next(arrays(obj), None)


def model_on(model, device):
    """model itself if it already lives on device, else its copy for device (weights copied once)."""
    first = next(itertools.chain(model.parameters(), model.buffers()), None)
    if first is None or first.device == device:
        return model
    # shortcut: a cached copy goes stale if the user changes the weights after first use (rare in inference-only
    # scope), key the cache on a weights version if fine-tuning ever comes into scope.
    with _models_lock:
        copies = _models.setdefault(model, {})
        if device not in copies:
            copies[device] = copy.deepcopy(model).to(device)
        return copies[device]


def _map(obj, fn):
    if isinstance(obj, dict):
        out = copy.copy(obj)  # keeps dict subclasses (defaultdict, OrderedDict) intact
        out.update((k, fn(v)) for k, v in obj.items())
        return out
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*map(fn, obj))
    if isinstance(obj, (list, tuple)):
        return type(obj)(map(fn, obj))
    return obj
