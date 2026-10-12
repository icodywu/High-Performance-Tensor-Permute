"""Tests of the hptpermute Python package against NumPy and (if installed) PyTorch. Run with: pytest tests"""
import itertools
import random

import numpy as np
import pytest

import hptpermute as hp

DTYPES = [np.uint8, np.int16, np.float32, np.float64, np.complex128, np.dtype("V3"), np.dtype("V12"), np.dtype("V24")]


def rand_array(shape, dtype, rng):
    dtype = np.dtype(dtype)
    raw = rng.integers(0, 256, size=int(np.prod(shape, dtype=np.int64)) * dtype.itemsize, dtype=np.uint8)
    return raw.view(dtype).reshape(shape)


def expect(x, dims):
    return np.transpose(x, dims).copy(order="C")  # (ascontiguousarray makes a 0-d array 1-d in NumPy 1.x)


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("dtype", DTYPES, ids=str)
def test_random(dtype):
    rng = np.random.default_rng(1)
    pyrng = random.Random(1)
    for _ in range(150):
        ndim = pyrng.randint(0, 6)
        shape = tuple(pyrng.choice([1, 1, 2, 3, 4, 5, 7, 16, 33]) for _ in range(ndim))
        dims = list(range(ndim))
        pyrng.shuffle(dims)
        x = rand_array(shape, dtype, rng)
        threads = pyrng.choice([None, 1, 2, 5])
        y = hp.permute(x, dims, num_threads=threads)
        assert y.flags.c_contiguous and same(y, expect(x, dims)), (shape, dims, threads)


def test_all_perms_4d():
    x = rand_array((3, 4, 5, 6), np.float32, np.random.default_rng(2))
    for dims in itertools.permutations(range(4)):
        assert same(hp.permute(x, dims), expect(x, dims))


def test_large_transposes():
    rng = np.random.default_rng(3)
    for shape, dims in [((1031, 517), (1, 0)), ((3, 257, 129), (0, 2, 1)), ((2, 67, 13, 11), (0, 2, 3, 1))]:
        for dtype in (np.uint8, np.float32, np.dtype("V6")):
            x = rand_array(shape, dtype, rng)
            for threads in (1, 4, None):
                assert same(hp.permute(x, dims, num_threads=threads), expect(x, dims))


def test_views_and_contiguous():
    rng = np.random.default_rng(4)
    base = rand_array((4, 5, 6, 7), np.int32, rng)
    view = base.transpose(2, 0, 3, 1)              # a permuted view: read in place
    assert same(hp.permute(view, (1, 0, 3, 2)), expect(view, (1, 0, 3, 2)))
    assert same(hp.contiguous(view), view.copy(order="C"))
    assert hp.contiguous(base) is base
    sliced = base[:, ::2, :, 1:]                   # not dense: copied first
    assert same(hp.permute(sliced, (3, 2, 1, 0)), expect(sliced, (3, 2, 1, 0)))
    flipped = base[::-1]
    assert same(hp.permute(flipped, (1, 0, 2, 3)), expect(flipped, (1, 0, 2, 3)))
    assert same(hp.permute(base, (-1, 0, 1, 2)), expect(base, (3, 0, 1, 2)))


def test_out_and_empty():
    x = rand_array((6, 7, 8), np.float64, np.random.default_rng(5))
    out = np.empty((8, 6, 7), np.float64)
    assert hp.permute(x, (2, 0, 1), out=out) is out and same(out, expect(x, (2, 0, 1)))
    e = np.zeros((3, 0, 2), np.float32)
    assert hp.permute(e, (2, 1, 0)).shape == (2, 0, 3)
    s = np.array(5.0)
    assert same(hp.permute(s, ()), s)


def test_errors():
    x = np.zeros((2, 3), np.float32)
    with pytest.raises(ValueError):
        hp.permute(x, (0, 0))
    with pytest.raises(ValueError):
        hp.permute(x, (0, 1, 2))
    with pytest.raises(ValueError):
        hp.permute(x, (1, 0), out=np.empty((2, 3), np.float32))
    with pytest.raises(ValueError):
        hp.permute(x, (1, 0), out=np.empty((3, 2), np.float64))
    with pytest.raises(ValueError):
        hp.permute(x, (1, 0), num_threads=0)
    with pytest.raises(TypeError):
        hp.permute([[1, 2]], (1, 0))
    with pytest.raises(TypeError):
        hp.permute(np.zeros((2, 2), object), (1, 0))
    buf = np.zeros(12, np.float32)
    with pytest.raises(hp.HptpError):
        hp.permute(buf[:6].reshape(2, 3), (1, 0), out=buf[:6].reshape(3, 2))


def test_info():
    assert hp.library_version() == hp.__version__
    assert hp.auto_threads(100) == 1 and hp.auto_threads(1 << 30) >= 1


try:
    import torch
except ImportError:
    torch = None


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_torch():
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16, torch.int64, torch.bool, torch.complex64, torch.uint8):
        x = torch.randint(0, 2 if dtype == torch.bool else 100, (5, 6, 7, 8), generator=g).to(dtype)
        for dims in [(0, 2, 3, 1), (3, 2, 1, 0), (1, 0, 2, 3), (0, 1, 2, 3)]:
            y = hp.permute(x, dims, num_threads=3)
            assert y.is_contiguous() and torch.equal(y, x.permute(dims).contiguous())
        v = x.transpose(1, 3)
        assert torch.equal(hp.contiguous(v), v.contiguous())
        assert torch.equal(hp.permute(v, (3, 1, 0, 2)), v.permute(3, 1, 0, 2).contiguous())
    z = torch.randn(4, 5, dtype=torch.complex64, generator=g).conj()
    assert torch.equal(hp.permute(z, (1, 0)), z.permute(1, 0).contiguous())
    out = torch.empty(8, 5, 6, 7)
    x = torch.randn(5, 6, 7, 8, generator=g)
    assert hp.permute(x, (3, 0, 1, 2), out=out) is out and torch.equal(out, x.permute(3, 0, 1, 2))
    with pytest.raises(ValueError):
        hp.permute(x, (3, 0, 1, 2), out=np.empty((8, 5, 6, 7), np.float32))
