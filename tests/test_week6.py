"""Week-6 exit check (DESIGN.md §11), plus the stress tools it relies on.

Run on a machine with an NVIDIA GPU:  pytest -m gpu tests/test_week6.py -s
"""
import json

import psutil
import pytest
import torch

import autohetero
from autohetero import calibrate, history, ops
from autohetero.ops import inputs
from bench import stress


def test_cpu_stress_starts_and_stops_its_processes():
    me = psutil.Process()
    with stress.cpu(workers=1) as tool:
        assert me.children(recursive=True) and tool
    assert not [p for p in me.children(recursive=True) if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]


@pytest.fixture
def calibrated(tmp_path_factory, monkeypatch):
    """One real calibration shared by this module's GPU tests (~30 s, once)."""
    monkeypatch.setattr(calibrate, "CACHE", tmp_path_factory.getbasetemp() / "week6-cache")
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)  # compare op code, not TF32 rounding
    autohetero.enable()


@pytest.mark.gpu
def test_exit_check_long_task_switches_to_hybrid_within_two_chunks_under_gpu_stress(calibrated):
    x = torch.rand(256, 3, 224, 224, generator=torch.Generator().manual_seed(0))
    ops.resnet18(x)  # idle: decides, and copies the model to the GPU
    with stress.gpu():
        ops.resnet18(x)
    history.flush()
    record = json.loads((calibrate.folder() / "history.jsonl").read_text().splitlines()[-1])
    assert record["mode"] == "gpu", f"idle decision was {record['mode']}: the check needs a GPU-chosen task"
    print(f"slowdown {record['slowdown']}x after {record['chunks_before_switch']} chunks; pieces {record['pieces']}")
    assert record["slowdown"] is not None and record["chunks_before_switch"] <= 2


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(ops.SPLITTABLE))
def test_exit_check_hybrid_results_match_single_device_results(calibrated, name):
    size = {"blur": (16, 3, 256, 256), "resize": (16, 3, 256, 256), "sobel": (16, 3, 256, 256),
            "resnet18": (16, 3, 224, 224), "mobilenetv2": (16, 3, 224, 224), "filter": 10**5,
            "matmul": 1024, "fft": (64, 4096)}[name]
    args = inputs.make(name, size, torch.Generator().manual_seed(0))
    results = {}
    for mode in ("hybrid", "cpu", "gpu"):
        with autohetero.force(mode):
            results[mode] = getattr(ops, name)(*args)
    for single in ("cpu", "gpu"):
        ref = results[single]
        torch.testing.assert_close(results["hybrid"], ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item())
