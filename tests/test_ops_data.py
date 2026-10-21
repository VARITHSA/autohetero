import numpy as np
import pytest
import torch

from autohetero import ops

rand = torch.Generator().manual_seed(0)


def test_filter_keeps_matching_rows_in_order():
    table = torch.tensor([[0.9, 1.0], [0.1, 2.0], [0.7, 3.0]])
    torch.testing.assert_close(ops.filter(table), torch.tensor([[0.9, 1.0], [0.7, 3.0]]))


def test_sort_is_stable_on_ties():
    table = torch.tensor([[1.0, 10.0], [0.0, 20.0], [1.0, 30.0]])
    torch.testing.assert_close(ops.sort(table), torch.tensor([[0.0, 20.0], [1.0, 10.0], [1.0, 30.0]]))


def test_groupby_sums_per_key():
    keys, sums = ops.groupby(torch.tensor([3, 1, 3, 7]), torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 1.0], [4.0, 0.0]]))
    assert keys.tolist() == [1, 3, 7]
    assert sums.tolist() == [[2.0, 0.0], [4.0, 1.0], [4.0, 0.0]]


def test_matmul_and_fft_match_numpy():
    a, b = torch.rand(5, 4, generator=rand), torch.rand(4, 3, generator=rand)
    np.testing.assert_allclose(ops.matmul(a, b).numpy(), a.numpy() @ b.numpy(), rtol=1e-5)
    signals = torch.rand(3, 64, generator=rand, dtype=torch.float64)
    np.testing.assert_allclose(ops.fft(signals).numpy(), np.fft.fft(signals.numpy()))


@pytest.mark.parametrize("op, args", [
    (ops.filter, (torch.rand(100, 3, generator=rand),)),
    (ops.matmul, (torch.rand(10, 6, generator=rand), torch.rand(6, 4, generator=rand))),
    (ops.fft, (torch.rand(10, 32, generator=rand),)),
])
def test_splittable_ops_split_on_their_first_argument(op, args):
    first, rest = args[0], args[1:]
    chunked = torch.cat([op(chunk, *rest) for chunk in first.split(3)])
    torch.testing.assert_close(chunked, op(*args))
