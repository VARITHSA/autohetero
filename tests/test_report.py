import pytest
import torch

import autohetero
from autohetero import dispatch, history, predictor


def record(name, mode, seconds, predicted_gpu=None, fallback=False, slowdown=None):
    history.record({"name": name, "op": "user", "elements": 10, "bytes": 40, "batch": 10, "dtype": "float32",
                    "cpu_load": None, "gpu_load": None, "mode": mode, "forced": predicted_gpu is None,
                    "predicted": {} if predicted_gpu is None else {"cpu": 1.0, "gpu": predicted_gpu},
                    "seconds": seconds, "pieces": [], "fallback": fallback, "slowdown": slowdown,
                    "chunks_before_switch": None})


def test_rows_totals_and_the_estimate_label(capsys):
    record("__main__.enhance", "gpu", 0.5, predicted_gpu=0.6)
    record("__main__.enhance", "hybrid", 0.4, predicted_gpu=0.6, slowdown=2.0)
    record("__main__.enhance", "gpu", 0.3, fallback=True)  # forced: no prediction, so no estimate
    record("blur", "cpu", 0.002)
    rows = autohetero.report()
    out = capsys.readouterr().out
    blur, enhance = rows
    assert enhance == {"function": "enhance", "calls": 3, "modes": "GPU 2, HYBRID 1", "seconds": pytest.approx(1.2),
                       "saved_vs_gpu_estimate": pytest.approx(0.3), "fallbacks": 1, "switches": 1}
    assert blur["saved_vs_gpu_estimate"] is None
    assert "saved vs GPU (est.)" in out and "ESTIMATE" in out and "n/a" in out and "total" in out


def test_empty_session(capsys):
    assert autohetero.report() == []
    assert "no dispatched calls yet" in capsys.readouterr().out


def test_dispatched_calls_show_up(monkeypatch, fake_gpu, capsys):
    monkeypatch.setattr(dispatch, "_unhealthy", set())
    monkeypatch.setattr(predictor, "decide", lambda name, f, splittable=False, quiet=False: ("gpu", {"cpu": 1.0, "gpu": 1.0}, False))
    double = autohetero.auto(lambda x: x * 2)
    for _ in range(3):
        double(torch.ones(4))
    (row,) = autohetero.report()
    assert row["calls"] == 3 and row["modes"] == "GPU 3" and row["saved_vs_gpu_estimate"] > 0
