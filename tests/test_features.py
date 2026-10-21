import math
import time

import numpy as np
import pytest
import torch

from autohetero import features
from autohetero.ops import inputs

LOAD = (12.5, None)


def test_numpy_and_torch_inputs_look_the_same():
    x = np.zeros((8, 3, 32, 32), dtype=np.float32)
    assert features.of("blur", (x,), LOAD) == features.of("blur", (torch.from_numpy(x),), LOAD) == {
        "op": "blur", "elements": 8 * 3 * 32 * 32, "bytes": 8 * 3 * 32 * 32 * 4, "batch": 8,
        "dtype": "float32", "cpu_load": 12.5, "gpu_load": None,
    }


def test_several_arrays_add_up_and_the_first_sets_batch_and_dtype():
    f = features.of("groupby", ({"keys": torch.zeros(100, dtype=torch.int64)}, [torch.zeros(100), "label"]), LOAD)
    assert (f["elements"], f["bytes"], f["batch"], f["dtype"]) == (200, 100 * 8 + 100 * 4, 100, "int64")


@pytest.mark.parametrize("args, batch_dim, batch", [
    ((torch.zeros(8, 3, 4, 4),), 1, 3),
    ((torch.zeros(8, 3),), -1, 3),
    ((torch.tensor(5.0),), 0, 1),  # 0-d: no batch dim
    ((5, "no arrays"), 0, 1),
])
def test_batch_size(args, batch_dim, batch):
    assert features.of("user", args, LOAD, batch_dim)["batch"] == batch


@pytest.mark.parametrize("op", inputs.OPS)
def test_dataset_rows_and_live_calls_get_the_same_features(op):
    size = inputs.sizes(op)[0][0]
    live = features.of(op, inputs.make(op, size), LOAD)
    assert features.of(op, inputs.make(op, size, device="meta"), LOAD) == live


def test_vector_matches_columns():
    v = features.vector(features.of("sort", (torch.zeros(10, 4),), LOAD))
    assert len(v) == len(features.COLUMNS)
    named = dict(zip(features.COLUMNS, v))
    assert named["bytes_per_element"] == 4 and named["cpu_load"] == 12.5 and math.isnan(named["gpu_load"])
    assert [c for c in features.COLUMNS if c.startswith("op=") and named[c]] == ["op=sort"]


def test_features_are_cheap_on_the_hot_path():
    args = (torch.zeros(64, 3, 224, 224),)
    start = time.perf_counter()
    for _ in range(10_000):
        features.vector(features.of("resnet18", args, LOAD))
    assert time.perf_counter() - start < 0.2  # under 20 us per call, against a < 1 ms dispatch budget
