import pytest
import torch

from autohetero import ops
from autohetero.ops import models

classifiers = pytest.mark.parametrize("op", [ops.resnet18, ops.mobilenetv2])


@classifiers
def test_logits_split_along_the_batch(op):
    images = torch.rand(4, 3, 64, 64, generator=torch.Generator().manual_seed(0))
    whole = op(images)
    assert whole.shape == (4, 1000)
    # eval mode: each sample is independent (train-mode BatchNorm would mix the batch)
    torch.testing.assert_close(torch.cat([op(half) for half in images.split(2)]), whole)


@classifiers
def test_weights_are_fixed_and_caller_rng_untouched(op):
    images = torch.ones(1, 3, 32, 32)
    before = op(images)
    models._models.clear()
    torch.manual_seed(123)
    expected_next = torch.rand(1)
    torch.manual_seed(123)
    torch.testing.assert_close(op(images), before)  # rebuilt from the same seed
    assert torch.equal(torch.rand(1), expected_next)
