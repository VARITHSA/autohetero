import time

import pytest
import torch

import autohetero
from autohetero import devices, dispatch, executor, features, history, predictor
from autohetero.ops import inputs


@pytest.fixture
def sleeper(monkeypatch, fake_gpu):
    """A user function whose speed per item on each device is set by the test; it logs (device, items)."""
    monkeypatch.setattr(predictor, "_cache", {})
    monkeypatch.setattr(predictor, "_modes", {})
    monkeypatch.setattr(dispatch, "_unhealthy", set())
    sizes = [64, 128, 256, 512]  # a tiny forest, so first calls have something to fall back on
    rows = [(features.of("matmul", inputs.make("matmul", n, device="meta"), (0.0, None)), d, 1e-9 * n * n)
            for n in sizes for d in ("cpu", "gpu")]
    predictor.train(rows, {"thread_seconds": 1e-5, "concat_bytes_per_s": 1e11})
    per_item, pieces = {"cpu": 1e-4, "gpu": 2e-3}, []

    def work(x):
        on = "gpu" if executor.running_on.get() == devices.GPU else "cpu"
        pieces.append((on, len(x)))
        time.sleep(per_item[on] * len(x))
        return x + 1

    return work, per_item, pieces


def test_first_splittable_call_is_a_trial_gpu_first_then_both_by_speed(sleeper, capsys):
    work, _, pieces = sleeper
    x = torch.arange(40.0)
    torch.testing.assert_close(autohetero.auto(splittable=True)(work)(x), x + 1)
    assert pieces[:2] == [("gpu", 4), ("cpu", 4)]  # ~10% each, GPU first
    assert sum(n for _, n in pieces) == 40
    assert "work: batch 40, TRIAL" in capsys.readouterr().err
    assert history.predict(f"{__name__}.{work.__qualname__}", "cpu", 40) is not None


def test_later_calls_follow_the_fits_and_change_mode_without_retraining(sleeper, capsys):
    work, per_item, pieces = sleeper
    auto_work = autohetero.auto(splittable=True)(work)
    x = torch.arange(10.0)
    auto_work(x)  # trial: the CPU is 20x faster here
    pieces.clear()
    auto_work(x)
    assert {on for on, _ in pieces} == {"cpu"} and "work: batch 10, CPU" in capsys.readouterr().err
    per_item["cpu"] = 5e-3  # now the CPU is 2.5x slower than the GPU
    for _ in range(10):
        pieces.clear()
        auto_work(x)
        if any(on == "gpu" for on, _ in pieces[:1]):  # the call now starts on the GPU (or in Hybrid)
            break
    assert any(on == "gpu" for on, _ in pieces), "the decision never moved off the slowed CPU"


def test_non_splittable_call_explores_the_other_device_once(sleeper):
    work, _, pieces = sleeper
    auto_work = autohetero.auto(work)
    for _ in range(4):
        auto_work(torch.arange(10.0))
    first, second, *rest = [on for on, _ in pieces]
    assert {first, second} == {"cpu", "gpu"}  # the forest's guess, then the other device once
    assert rest == ["cpu", "cpu"]  # then the fits: the CPU is 20x faster


def test_every_new_size_is_logged_even_after_its_bucket_was_a_chunk(sleeper, capsys):
    work, _, _ = sleeper
    auto_work = autohetero.auto(splittable=True)(work)
    auto_work(torch.arange(100.0))  # trial; its chunks ask about smaller sizes
    auto_work(torch.arange(100.0))
    capsys.readouterr()
    auto_work(torch.arange(10.0))  # the size of a chunk above, but a new decision for a real call
    assert "work: batch 10," in capsys.readouterr().err
