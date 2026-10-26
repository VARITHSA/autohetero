from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

DEMO = Path(__file__).resolve().parent.parent / "demo"


def test_demo_runs_the_pipeline_three_ways(monkeypatch):
    from autohetero import dispatch

    monkeypatch.setattr(dispatch, "_enabled", dispatch._enabled)  # the app calls enable(): undo it afterwards
    at = AppTest.from_file(str(DEMO / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    at.select_slider[0].set_value(1)
    at.selectbox[0].set_value("256 x 256")
    at.button[0].click()
    at.run()
    assert not at.exception
    assert [m.label for m in at.metric] == ["AutoHetero", "CPU only", "GPU only"]
    steps = at.dataframe[0].value
    assert list(steps.step) == ["Blur", "Edges", "Shrink", "Classify"]
    assert len(at.dataframe) == 4  # three step tables and the report


def test_the_pipeline_is_plain_pytorch_plus_one_line():
    from eval import rq3

    assert rq3.lines_added(DEMO / "pipeline.py") == ["autohetero.enable()  # <- the one added line"]


def test_playground_runs_an_op_every_way_and_shows_the_snippet(monkeypatch, fake_gpu):
    from autohetero import calibrate, dispatch, predictor

    monkeypatch.setattr(dispatch, "_enabled", dispatch._enabled)  # the page calls enable(): undo it afterwards
    monkeypatch.setattr(calibrate, "calibration", lambda force=False: {"seconds": 1.0, "h2d_bytes_per_s": 6e9, "d2h_bytes_per_s": 3e9})
    monkeypatch.setattr(predictor, "decide", lambda name, f, splittable=False, quiet=False: ("gpu", {"cpu": 2e-3, "gpu": 1e-3}, False))
    at = AppTest.from_file(str(DEMO / "pages" / "playground.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.selectbox[0].set_value("matmul")
    at.run()
    at.select_slider[0].set_value(0)  # the smallest size: 64 x 64
    at.button[0].click()
    at.run()
    assert not at.exception
    assert list(at.dataframe[0].value.option) == ["AutoHetero", "GPU only", "Hybrid", "CPU only"]
    assert "out = autohetero.ops.matmul(a, b)" in at.code[0].value


def test_playground_without_a_gpu_offers_the_cpu_only(monkeypatch):
    from autohetero import devices, dispatch

    monkeypatch.setattr(dispatch, "_enabled", dispatch._enabled)
    monkeypatch.setattr(devices, "GPU", None)
    at = AppTest.from_file(str(DEMO / "pages" / "playground.py"), default_timeout=60)
    at.run()
    assert "No NVIDIA GPU" in at.warning[0].value
    at.selectbox[0].set_value("sort")
    at.run()
    at.select_slider[0].set_value(0)  # the smallest size: 1,000 rows
    at.button[0].click()
    at.run()
    assert not at.exception
    assert list(at.dataframe[0].value.option) == ["AutoHetero", "CPU only"]  # sort can't split: no Hybrid either
