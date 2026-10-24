import json
import time

import numpy as np
import pytest
import torch

import autohetero
from autohetero import calibrate, devices, dispatch, executor, history, ops, predictor
from autohetero.ops import inputs


@pytest.fixture
def predicts(monkeypatch, fake_gpu):
    """Make the predictor answer mode, with cpu and gpu seconds per batch item, for every call or piece."""
    def use(mode, cpu, gpu):
        decide = lambda name, f, splittable=False, quiet=False: (mode, {"cpu": cpu * f["batch"], "gpu": gpu * f["batch"]}, False)
        monkeypatch.setattr(predictor, "decide", decide)
    monkeypatch.setattr(dispatch, "_unhealthy", set())
    monkeypatch.setattr(dispatch, "_enabled", True)
    return use


def recorder():
    """An @auto-able function that doubles x and logs (device, rows) for each piece it runs."""
    pieces = []

    def double(x):
        pieces.append((executor.running_on.get(), len(x)))
        return x * 2

    return double, pieces


SMALL = {"blur": (8, 3, 32, 32), "resize": (8, 3, 32, 32), "sobel": (8, 3, 32, 32), "resnet18": (8, 3, 64, 64),
         "mobilenetv2": (8, 3, 64, 64), "filter": 1000, "matmul": 64, "fft": (16, 1024)}


@pytest.mark.parametrize("name", sorted(ops.SPLITTABLE))
def test_hybrid_output_equals_single_device_output(predicts, name):
    predicts("hybrid", 1.0, 1.0)  # equal speeds: each Hybrid round splits half and half
    args = inputs.make(name, SMALL[name], torch.Generator().manual_seed(0))
    hybrid, single = getattr(ops, name)(*args), ops.RAW[name](*args)
    torch.testing.assert_close(hybrid, single, rtol=1e-4, atol=1e-4 * single.abs().max().item())


def test_hybrid_uses_both_devices_by_speed_and_keeps_order(predicts):
    predicts("hybrid", 3.0, 1.0)  # GPU 3x faster: ~75% of each round
    double, pieces = recorder()
    x = np.arange(40, dtype=np.float32).reshape(20, 2)
    out = autohetero.auto(splittable=True)(double)(x)
    assert isinstance(out, np.ndarray) and (out == x * 2).all()
    rows = {d: sum(n for dev, n in pieces if dev == d) for d in (devices.GPU, devices.CPU)}
    assert rows[devices.GPU] + rows[devices.CPU] == 20 and rows[devices.GPU] > 2 * rows[devices.CPU] > 0


def test_user_functions_split_every_array_argument(predicts):
    predicts("hybrid", 1.0, 1.0)
    add = autohetero.auto(splittable=True)(lambda x, y, scale=1: (x + y) * scale)
    x, y = torch.arange(10.0), torch.arange(10.0) * 10
    torch.testing.assert_close(add(x, y=y, scale=2), (x + y) * 2)


def test_slowdown_switches_to_hybrid_within_two_chunks(predicts, monkeypatch, capsys):
    per_item = 0.001
    predicts("gpu", per_item, per_item)  # predicted: 1 ms per row on either device
    pieces = []

    def slow_on_gpu(x):  # the GPU is 4x slower than predicted, as under contention
        on = executor.running_on.get()
        pieces.append((on, len(x)))
        time.sleep(len(x) * per_item * (4 if on == devices.GPU else 1))
        return x + 1

    x = torch.arange(100.0)
    torch.testing.assert_close(autohetero.auto(splittable=True)(slow_on_gpu)(x), x + 1)
    history.flush()
    record = json.loads((calibrate.folder() / "history.jsonl").read_text().splitlines()[-1])
    assert record["chunks_before_switch"] == 2  # the first chunk isn't judged; the second is
    assert record["slowdown"] >= 3  # 4x by construction; sleep() can only overshoot
    assert "slow_on_gpu: slowdown" in capsys.readouterr().err
    on_cpu = sum(n for on, n in pieces if on == devices.CPU)
    assert on_cpu > 50  # after the switch most rows go to the now-faster CPU


def test_non_splittable_functions_are_never_chunked(predicts):
    predicts("gpu", 0.2, 0.1)  # 10-20 s: long enough to chunk, if it were allowed
    double, pieces = recorder()
    autohetero.auto(double)(torch.zeros(100))
    with autohetero.force("hybrid"):
        autohetero.auto(double)(torch.zeros(100))
    assert pieces == [(devices.GPU, 100), (devices.GPU, 100)]


def test_short_calls_and_forced_single_devices_run_in_one_piece(predicts):
    predicts("gpu", 1e-5, 1e-5)  # 1 ms for 100 rows: shorter than one 5 ms chunk
    double, pieces = recorder()
    split_double = autohetero.auto(splittable=True)(double)
    split_double(torch.zeros(100))
    predicts("gpu", 0.1, 0.1)
    with autohetero.force("gpu"):
        split_double(torch.zeros(100))
    assert pieces == [(devices.GPU, 100), (devices.GPU, 100)]


def test_a_gpu_failure_inside_hybrid_reruns_the_whole_call_on_the_cpu(predicts, capsys):
    predicts("hybrid", 1.0, 1.0)
    pieces = []

    def fragile(x):
        pieces.append((executor.running_on.get(), len(x)))
        if executor.running_on.get() == devices.GPU:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        return x * 3

    x = torch.arange(8.0)
    torch.testing.assert_close(autohetero.auto(splittable=True)(fragile)(x), x * 3)
    assert pieces[-1] == (devices.CPU, 8) and "GPU failed" in capsys.readouterr().err


def test_join_rejects_outputs_it_cannot_join():
    with pytest.raises(TypeError, match="must return arrays"):
        executor._join([1, 2], 0)
    a = executor._join([(torch.ones(1), {"k": torch.zeros(1)}), (torch.ones(2), {"k": torch.zeros(2)})], 0)
    assert a[0].shape == (3,) and a[1]["k"].shape == (3,)


def test_short_hybrid_calls_run_in_one_round(predicts):
    predicts("hybrid", 1e-4, 1e-4)  # 20 rows: 1 ms on both devices together, below the 5 ms minimum round
    double, pieces = recorder()
    out = autohetero.auto(splittable=True)(double)(torch.arange(20.0))
    torch.testing.assert_close(out, torch.arange(20.0) * 2)
    assert sorted(pieces, key=str) == sorted([(devices.GPU, 10), (devices.CPU, 10)], key=str)  # one round, half each
