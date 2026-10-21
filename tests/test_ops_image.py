import pytest
import torch

from autohetero import ops


@pytest.mark.parametrize("shape", [(5, 6), (3, 5, 6), (2, 3, 5, 6)])
def test_blur_keeps_shape_and_constant_images(shape):
    out = ops.blur(torch.full(shape, 0.7))
    assert out.shape == shape
    torch.testing.assert_close(out, torch.full(shape, 0.7))


def test_blur_smooths_noise():
    noise = torch.rand(1, 1, 64, 64, generator=torch.Generator().manual_seed(0))
    assert ops.blur(noise).std() < noise.std() / 3


def test_resize_halves_and_keeps_constants():
    out = ops.resize(torch.full((2, 3, 9, 16), 0.3))
    assert out.shape == (2, 3, 4, 8)
    torch.testing.assert_close(out, torch.full(out.shape, 0.3))


def test_sobel_finds_a_vertical_edge_only():
    step = torch.zeros(1, 1, 8, 8)
    step[..., 4:] = 1.0
    out = ops.sobel(step)
    expected = torch.zeros_like(step)
    expected[..., 3:5] = 4.0  # 1 + 2 + 1 for a unit step, on the two columns either side of it
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(ops.sobel(torch.ones(3, 8, 8)), torch.zeros(3, 8, 8))
