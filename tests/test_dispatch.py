import threading

import numpy as np
import pytest
import torch

import autohetero
from autohetero import devices, dispatch, executor, ops, predictor


@pytest.fixture
def decides(monkeypatch, fake_gpu):
    """Make the predictor answer `mode`, recording each decision it is asked for."""
    asked = []

    def use(mode):
        def decide(name, f, splittable=False, quiet=False):
            asked.append((name, f["op"], splittable))
            return mode, {"cpu": 1.0, "gpu": 2.0}, True
        monkeypatch.setattr(predictor, "decide", decide)
        return asked

    monkeypatch.setattr(dispatch, "_unhealthy", set())
    monkeypatch.setattr(dispatch, "_enabled", False)
    return use


def ran_on():
    return executor.running_on.get()


def test_numpy_in_numpy_out_and_the_function_sees_torch_on_the_gpu(decides):
    decides("gpu")
    seen = []

    @autohetero.auto
    def double(x):
        seen.append((type(x), ran_on()))
        return x * 2

    out = double(np.arange(4, dtype=np.float32))
    assert isinstance(out, np.ndarray) and out.tolist() == [0, 2, 4, 6]
    assert seen == [(torch.Tensor, devices.GPU)]


def test_decision_log_once_per_new_decision(decides, capsys, monkeypatch):
    decides("gpu")
    auto_fn = autohetero.auto(lambda x: x)
    auto_fn(torch.zeros(200, 3))
    assert capsys.readouterr().err == "[AutoHetero] <lambda>: batch 200, GPU\n"
    monkeypatch.setenv("AUTOHETERO_QUIET", "1")
    auto_fn(torch.zeros(200, 3))
    assert capsys.readouterr().err == ""


def test_force_beats_the_prediction_and_stays_in_its_own_thread(decides):
    asked = decides("cpu")
    seen = {}
    both_inside = threading.Barrier(2)

    @autohetero.auto
    def where(x, label):
        both_inside.wait(timeout=5)  # the two threads overlap inside the call
        seen[label] = ran_on()
        return x

    def forced():
        with autohetero.force("gpu"):
            where(torch.zeros(2), "forced")

    threads = [threading.Thread(target=forced), threading.Thread(target=where, args=(torch.zeros(2), "free"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"forced": devices.GPU, "free": devices.CPU}
    assert len(asked) == 1  # only the unforced call was predicted
    with pytest.raises(ValueError):
        with autohetero.force("tpu"):
            pass


def test_gpu_error_reruns_on_cpu_and_that_size_stays_on_cpu(decides, capsys):
    decides("gpu")
    attempts = []

    @autohetero.auto
    def fragile(x):
        attempts.append(ran_on())
        if ran_on() == devices.GPU:
            raise RuntimeError("CUDA error: out of memory\nCUDA kernel errors might be asynchronously reported")
        return x + 1

    assert fragile(torch.zeros(8)).tolist() == [1.0] * 8
    assert attempts == [devices.GPU, devices.CPU]
    assert "fragile: GPU failed (RuntimeError: CUDA error: out of memory); this size runs on the CPU" in capsys.readouterr().err
    fragile(torch.zeros(9))  # same log2 size bucket: straight to the CPU
    assert attempts[2:] == [devices.CPU]
    fragile(torch.zeros(64))  # another size: the GPU is tried again
    assert attempts[3:] == [devices.GPU, devices.CPU]


def test_the_functions_own_errors_still_reach_the_caller(decides):
    decides("gpu")

    @autohetero.auto
    def buggy(x):
        raise RuntimeError("shape mismatch")

    with pytest.raises(RuntimeError, match="shape mismatch"):
        buggy(torch.zeros(4))
    assert dispatch._unhealthy == set()  # it failed on the CPU too, so the GPU isn't blamed


def test_builtin_op_inside_auto_runs_on_the_outer_device_without_a_new_decision(decides, monkeypatch):
    asked = decides("gpu")
    inner = []
    raw_sort = ops.RAW["sort"]
    monkeypatch.setitem(ops.RAW, "sort", lambda t: inner.append(ran_on()) or raw_sort(t))
    monkeypatch.setattr(ops, "sort", ops._dispatched("sort"))

    @autohetero.auto
    def pipeline(table):
        return ops.sort(table)

    out = pipeline(np.array([[2.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    assert isinstance(out, np.ndarray) and out[:, 0].tolist() == [1.0, 2.0]
    assert inner == [devices.GPU] and [a[0] for a in asked] == [f"{__name__}.test_builtin_op_inside_auto_runs_on_the_outer_device_without_a_new_decision.<locals>.pipeline"]


def test_builtin_ops_dispatch_only_after_enable(decides, monkeypatch):
    asked = decides("cpu")
    table = torch.rand(10, 4)
    ops.sort(table)
    assert asked == []
    monkeypatch.setattr(dispatch, "_enabled", True)
    ops.sort(table)
    ops.matmul(table, table.T)
    assert list(dict.fromkeys(asked)) == [("sort", "sort", False), ("matmul", "matmul", True)]  # matmul also asks per chunk


def test_calls_without_arrays_run_as_written(decides):
    asked = decides("gpu")
    assert autohetero.auto(lambda a, b: a + b)(2, 3) == 5
    assert asked == []


def test_no_gpu_runs_everything_on_the_cpu(monkeypatch, capsys):
    monkeypatch.setattr(devices, "GPU", None)
    monkeypatch.setattr(dispatch, "_said", set())
    seen = []
    where = autohetero.auto(lambda x: seen.append(ran_on()) or x)
    where(np.zeros(3))
    with autohetero.force("gpu"):
        where(np.zeros(3))
    assert seen == [devices.CPU, devices.CPU]
    assert capsys.readouterr().err.count("no usable CUDA GPU: everything runs on the CPU") == 1  # said once


@pytest.mark.gpu
def test_real_gpu_numpy_round_trip_and_oom_fallback(monkeypatch):
    monkeypatch.setattr(dispatch, "_unhealthy", set())

    @autohetero.auto
    def greedy(x):
        if x.is_cuda:
            torch.empty(2**50, device=x.device)  # 4 PiB: out of memory on any GPU
        return x * 2

    with autohetero.force("gpu"):
        assert greedy(np.ones(3, dtype=np.float32)).tolist() == [2.0, 2.0, 2.0]
        assert autohetero.auto(lambda x: x.device.type)(np.ones(3)) == "cuda"


def test_a_gpu_failure_while_calibrating_never_reaches_the_caller(monkeypatch, fake_gpu, capsys):
    def broken():
        raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 64.00 MiB")

    monkeypatch.setattr(dispatch.calibrate, "calibration", broken)
    monkeypatch.setattr(predictor, "_model", None)
    monkeypatch.setattr(predictor, "_cache", {})
    monkeypatch.setattr(dispatch, "_said", set())
    monkeypatch.setattr(dispatch, "_enabled", dispatch._enabled)  # enable() below: undo it afterwards
    autohetero.enable()
    assert devices.GPU is None  # (monkeypatch restores it after the test)
    assert autohetero.auto(lambda x: x + 1)(torch.zeros(2)).tolist() == [1.0, 1.0]
    assert capsys.readouterr().err.count("GPU unusable (OutOfMemoryError") == 1
