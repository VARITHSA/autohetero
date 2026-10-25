import csv
import time

import numpy as np
import pytest
import torch

from autohetero import calibrate, devices, ops
from autohetero.ops import inputs


@pytest.fixture
def quick(tmp_path, monkeypatch):
    """Calibration into a temporary cache, on a 2 s budget, without log output."""
    monkeypatch.setattr(calibrate, "CACHE", tmp_path)
    monkeypatch.setattr(calibrate, "BUDGET", 2.0)
    monkeypatch.setenv("AUTOHETERO_QUIET", "1")


def test_no_gpu_means_no_calibration(monkeypatch):
    monkeypatch.setattr(devices, "GPU", None)
    assert calibrate.calibration() is None


def test_calibrates_every_op_once_per_machine(quick, fake_gpu, monkeypatch):
    info = calibrate.calibration()
    with open(info["timings"]) as f:
        rows = list(csv.DictReader(f))
    for op in inputs.OPS:
        for device in ("cpu", "gpu"):
            sizes = {r["size"] for r in rows if r["op"] == op and r["device"] == device}
            assert 1 <= len(sizes) <= calibrate.SIZES_PER_OP, (op, device)
    assert info["h2d_bytes_per_s"] > 0 and info["launch_seconds"] > 0 and info["thread_seconds"] > 0
    assert info["seconds"] < calibrate.BUDGET + 3  # slack: the first size of each op always runs

    start = time.perf_counter()
    assert calibrate.calibration() == info  # cached
    assert time.perf_counter() - start < 0.5

    monkeypatch.setattr(calibrate, "machine", lambda: {**info["machine"], "torch": "other"})
    assert calibrate.calibration()["machine"]["torch"] == "other"  # new machine key: calibrated again


def test_time_call_discards_warmup_and_measures_the_call():
    calls = []

    def slow_identity(x):
        calls.append(1)
        time.sleep(0.01)
        return x

    times = calibrate.time_call(slow_identity, (np.ones(3),), devices.CPU, reps=3, warmup=2)
    assert len(calls) == 5 and len(times) == 3
    assert all(0.01 <= t < 0.5 for t in times)


@pytest.mark.gpu
def test_real_calibration(tmp_path, monkeypatch):
    monkeypatch.setattr(calibrate, "CACHE", tmp_path)
    info = calibrate.calibration()
    print(f"calibration took {info['seconds']} s")  # reported, not asserted: ~30 s is a design budget
    assert info["machine"]["gpu"] and info["h2d_bytes_per_s"] > 0
    with open(info["timings"]) as f:
        assert {r["device"] for r in csv.DictReader(f)} == {"cpu", "gpu"}


@pytest.mark.gpu
def test_time_call_on_real_gpu():
    a = torch.rand(1024, 1024, generator=torch.Generator().manual_seed(0))
    assert all(t > 0 for t in calibrate.time_call(ops.RAW["matmul"], (a, a), devices.GPU))


def test_next_run_time_scales_with_the_measured_growth_rate():
    assert calibrate._predicted([(1e3, 1.0)], 1e5) == pytest.approx(1000)  # default growth 1.5
    assert calibrate._predicted([(1e3, 1.0), (1e4, 10.0)], 1e5) == pytest.approx(100)  # measured: linear
    assert calibrate._predicted([(1e3, 1.0), (1e4, 0.5)], 1e5) == pytest.approx(5)  # noise can't predict a speed-up
    assert calibrate._predicted([(1e3, 1.0), (1e4, 1e3)], 1e5) == pytest.approx(1e5)  # capped at quadratic


def test_an_op_with_no_size_that_fits_is_skipped_not_a_crash(monkeypatch):
    monkeypatch.setattr(calibrate, "fits", lambda op, size, device: False)
    rows = []
    assert calibrate._sweep("matmul", {"cpu": devices.CPU}, time.perf_counter() + 1, rows) == {"cpu": 0}
    assert rows == []
