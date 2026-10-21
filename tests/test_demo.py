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
