"""Fast, multi-threaded permutation (transpose) of dense NumPy arrays and PyTorch CPU tensors.

    import hptpermute as hp
    y = hp.permute(x, (0, 2, 3, 1))      # == x.permute(0, 2, 3, 1).contiguous()  /  np.ascontiguousarray(x.transpose(...))
    y = hp.contiguous(x.transpose(1, 2)) # == x.transpose(1, 2).contiguous(), for a permuted view
    hp.permute(x, (1, 0), out=y)         # into an existing contiguous array or tensor

The work is done by the C++ library of https://github.com/icodywu/High-Performance-Tensor-Permute, called through
ctypes with the GIL released.
"""
import ctypes
import glob
import importlib.machinery
import os

__all__ = ["permute", "contiguous", "auto_threads", "library_version", "HptpError"]
__version__ = "1.0.0"


class HptpError(RuntimeError):
    """An error reported by the C library."""


def _load():
    path = os.environ.get("HPTP_LIBRARY")
    if not path:
        here = os.path.dirname(os.path.abspath(__file__))
        found = [p for suffix in importlib.machinery.EXTENSION_SUFFIXES for p in glob.glob(os.path.join(here, "_hptp" + suffix))]
        if not found:
            raise ImportError("hptpermute: the compiled library _hptp is missing; install the package with 'pip install .'")
        path = found[0]
    lib = ctypes.CDLL(path)
    i64p = ctypes.POINTER(ctypes.c_int64)
    lib.hptp_permute.restype = ctypes.c_int
    lib.hptp_permute.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, i64p, i64p, ctypes.c_int]
    lib.hptp_auto_threads.restype = ctypes.c_int
    lib.hptp_auto_threads.argtypes = [ctypes.c_size_t]
    lib.hptp_strerror.restype = ctypes.c_char_p
    lib.hptp_strerror.argtypes = [ctypes.c_int]
    lib.hptp_version.restype = ctypes.c_char_p
    lib.hptp_version.argtypes = []
    return lib


_lib = _load()
MAX_DIMS = 64
_I64 = [ctypes.c_int64 * max(n, 1) for n in range(MAX_DIMS + 1)]  # argument array types, by ndim
_torch = None


def _torch_module():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def library_version():
    """Version of the loaded C library."""
    return _lib.hptp_version().decode()


def auto_threads(nbytes):
    """Number of threads that num_threads=None uses for a tensor of nbytes bytes."""
    return _lib.hptp_auto_threads(int(nbytes))


class _View:
    """What the C call needs from an array or tensor: shape, strides in elements (None if not whole elements),
    element size, data address, and how to make a contiguous copy or a new array."""

    def __init__(self, x):
        if type(x).__module__.startswith("torch"):
            torch = _torch_module()
            if x.layout != torch.strided or x.device.type != "cpu" or x.is_quantized:
                raise TypeError("hptpermute: only dense (strided) CPU tensors are supported")
            if x.is_conj() or x.is_neg():
                x = x.resolve_conj().resolve_neg()
            self.kind = "torch"
            self.shape = tuple(x.shape)
            self.strides = tuple(x.stride())
            self.itemsize = x.element_size()
        elif hasattr(x, "__array_interface__") and hasattr(x, "ctypes") and hasattr(x, "itemsize"):
            if x.dtype.hasobject:
                raise TypeError("hptpermute: arrays of Python objects are not supported")
            self.kind = "numpy"
            self.shape = tuple(x.shape)
            self.itemsize = x.itemsize
            self.strides = tuple(s // x.itemsize for s in x.strides) if all(s % x.itemsize == 0 for s in x.strides) else None
        else:
            raise TypeError("hptpermute: expected a numpy.ndarray or a torch.Tensor, got %s" % type(x).__name__)
        self.x = x

    def address(self):
        return self.x.data_ptr() if self.kind == "torch" else self.x.ctypes.data

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n

    def make_contiguous(self):
        if self.kind == "torch":
            return _View(self.x.contiguous())
        import numpy as np
        return _View(np.ascontiguousarray(self.x))

    def empty(self, shape):
        if self.kind == "torch":
            return _torch_module().empty(shape, dtype=self.x.dtype)
        import numpy as np
        return np.empty(shape, dtype=self.x.dtype)

    def is_contiguous(self):
        if self.kind == "torch":
            return self.x.is_contiguous()
        return self.x.flags.c_contiguous


def _dense_order(shape, strides):
    """If the array is a permuted view of a dense C-order block (no gaps, no overlap, positive strides), the order of
    its dimensions in memory, outermost first; otherwise None."""
    if strides is None:
        return None
    big = [d for d in range(len(shape)) if shape[d] != 1]
    order = sorted(big, key=lambda d: -strides[d])
    expect = 1
    for d in reversed(order):
        if strides[d] != expect:
            return None
        expect *= shape[d]
    return [d for d in range(len(shape)) if shape[d] == 1] + order


def _normalize_dims(dims, ndim):
    dims = [int(d) for d in dims]
    if len(dims) != ndim:
        raise ValueError("hptpermute: dims has %d entries for a tensor of %d dimensions" % (len(dims), ndim))
    dims = [d + ndim if d < 0 else d for d in dims]
    if sorted(dims) != list(range(ndim)):
        raise ValueError("hptpermute: dims %s is not a permutation of the %d dimensions" % (tuple(dims), ndim))
    return dims


def _threads(num_threads):
    if num_threads is None:
        return 0
    n = int(num_threads)
    if n < 1:
        raise ValueError("hptpermute: num_threads must be at least 1 (or None for automatic)")
    return n


def permute(x, dims, *, out=None, num_threads=None):
    """Return x with its dimensions permuted by dims, as a new contiguous array or tensor of the same kind and dtype.

    Same values as x.permute(*dims).contiguous() for a torch.Tensor and np.ascontiguousarray(np.transpose(x, dims))
    for a numpy.ndarray. x may be any CPU array or tensor: a permuted view of a contiguous one (such as x.T) is read in
    place, anything else is first copied to contiguous memory. If out is given, it must be a contiguous array or
    tensor of the result's shape and x's dtype, not overlapping x; it is filled and returned. num_threads=None picks
    the number of threads from the tensor size (see auto_threads()); a number is an upper bound, as no call uses more
    than one thread per 64 KiB of tensor. Threads are kept in a pool between calls.
    """
    v = _View(x)
    ndim = len(v.shape)
    if ndim > MAX_DIMS:
        raise ValueError("hptpermute: at most %d dimensions are supported" % MAX_DIMS)
    dims = _normalize_dims(dims, ndim)
    out_shape = tuple(v.shape[d] for d in dims)
    if out is None:
        out = v.empty(out_shape)
        o = _View(out)
    else:
        o = _View(out)
        if (o.kind != v.kind or out.dtype != x.dtype or o.shape != out_shape or not o.is_contiguous()
                or (o.kind == "numpy" and not out.flags.writeable)):
            raise ValueError("hptpermute: out must be a writable contiguous %s of shape %s and dtype %s"
                             % ("tensor" if v.kind == "torch" else "array", out_shape, x.dtype))
    if v.numel() == 0:
        return out
    order = _dense_order(v.shape, v.strides)
    if order is None:
        v = v.make_contiguous()
        order = list(range(ndim))
    pos = [0] * ndim
    for k, d in enumerate(order):
        pos[d] = k
    base_shape = _I64[ndim](*[v.shape[d] for d in order])
    perm = _I64[ndim](*[pos[d] for d in dims])
    rc = _lib.hptp_permute(v.address(), o.address(), v.itemsize, ndim, base_shape, perm, _threads(num_threads))
    if rc != 0:
        raise HptpError("hptp_permute: %s (%d)" % (_lib.hptp_strerror(rc).decode(), rc))
    return out


def contiguous(x, *, num_threads=None):
    """Return x if it is contiguous, else a contiguous copy, like torch.Tensor.contiguous() and
    np.ascontiguousarray(). A permuted view (x.T, x.transpose(1, 2), x.permute(...)) is copied with the fast kernel."""
    v = _View(x)
    if v.is_contiguous():
        return x
    return permute(x, range(len(v.shape)), num_threads=num_threads)
