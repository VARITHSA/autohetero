import pytest
import torch


def pytest_collection_modifyitems(config, items):
    # pytest must pass on CPU-only machines (DESIGN.md §12); real-GPU tests run on Colab / the fixed machine.
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="no CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fake_gpu(monkeypatch):
    """cpu:0 stands in for the GPU: it computes on the CPU, but moving to it copies and it compares unequal
    to devices.CPU, so tests can tell which device a call ran on."""
    from autohetero import devices

    monkeypatch.setattr(devices, "GPU", torch.device("cpu", 0))


@pytest.fixture(autouse=True)
def private_cache(tmp_path, monkeypatch):
    """Every test gets its own machine cache and an empty call history: nothing touches ~/.cache."""
    from autohetero import calibrate, history

    monkeypatch.setattr(calibrate, "CACHE", tmp_path / "cache")
    history.reset()
    yield
    history.reset()
