import collections

import numpy as np
import pytest
import torch
from torch import nn

from autohetero import devices


def test_numpy_in_numpy_out(fake_gpu):
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    moved = devices.to_device(x, devices.GPU)
    assert isinstance(moved, torch.Tensor)
    out = devices.restore(moved * 2, like=x)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    np.testing.assert_array_equal(out, x * 2)


def test_containers_keep_their_shape():
    Pair = collections.namedtuple("Pair", "a b")
    x = np.ones(3)
    moved = devices.to_device({"p": Pair(x, 5), "l": [x, "s"]}, devices.CPU)
    assert isinstance(moved["p"], Pair) and moved["p"].b == 5
    assert isinstance(moved["l"][0], torch.Tensor) and moved["l"][1] == "s"
    counts = devices.to_device(collections.defaultdict(int, k=x), devices.CPU)
    assert isinstance(counts["k"], torch.Tensor) and counts["missing"] == 0
    assert devices.find_array(([], {"k": x})) is x
    assert devices.find_array((1, "s")) is None


def test_awkward_arrays_pass_through():
    reversed_view = np.arange(6.0)[::-1]
    np.testing.assert_array_equal(devices.to_device(reversed_view, devices.CPU).numpy(), reversed_view)
    assert devices.to_device(np.array(3.0), devices.CPU).shape == ()
    for no_torch_dtype in (np.array(["a", None], dtype=object), np.arange(3, dtype=">f4")):
        assert devices.to_device(no_torch_dtype, devices.CPU) is no_torch_dtype


def test_model_copied_once_per_device_and_never_moved():
    model = nn.Linear(2, 2)
    meta = torch.device("meta")  # a second device that exists on every machine
    first = devices.model_on(model, meta)
    assert first is not model and first.weight.device == meta
    assert devices.model_on(model, meta) is first
    assert devices.model_on(model, devices.CPU) is model
    assert model.weight.device == devices.CPU


def test_tensor_outputs_go_back_to_the_input_device():
    meta = torch.device("meta")
    assert devices.restore((torch.ones(2), "s"), like=torch.empty(2, device=meta))[0].device == meta


@pytest.mark.gpu
def test_round_trip_on_real_gpu():
    x = np.random.default_rng(0).random((4, 4), dtype=np.float32)
    on_gpu = devices.to_device(x, devices.GPU)
    assert on_gpu.is_cuda
    np.testing.assert_array_equal(devices.restore(on_gpu, like=x), x)
    t = torch.ones(3)
    assert devices.restore(devices.to_device(t, devices.GPU), like=t).device == devices.CPU
    model = nn.Linear(2, 2)
    assert devices.model_on(model, devices.GPU).weight.is_cuda and not model.weight.is_cuda
