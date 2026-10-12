# High-Performance Tensor Permute

Fast, multi-threaded permutation (transpose) of dense tensors on the CPU: the work behind
`x.permute(dims).contiguous()` in PyTorch and `np.ascontiguousarray(x.transpose(dims))` in NumPy. It is one C++17
header, a small C library, and a Python package for NumPy arrays and PyTorch tensors.

```python
import hptpermute as hp

y = hp.permute(x, (0, 2, 3, 1))   # same values as x.permute(0, 2, 3, 1).contiguous(), as a new tensor
```

## Performance

Against PyTorch 2.14.1 on the CPU, in GB/s of data read plus written (higher is better). Each cell is PyTorch → this
library (speedup).

| Permutation | Shape | dtype | 1 thread | 8 threads | 32 threads |
|---|---|---|---:|---:|---:|
| 2-D transpose | 16384×16384 | float32 | 7.5 → 11.1 (1.5×) | 7.5 → 62.0 (8.3×) | 7.6 → 136 (18×) |
| 2-D transpose | 32768×32768 | uint8 | 3.0 → 4.8 (1.6×) | 3.0 → 37.3 (12×) | 3.0 → 86.0 (29×) |
| batched transpose | 64×1024×2048 | float32 | 2.3 → 13.7 (5.8×) | 18.4 → 63.2 (3.4×) | 59.1 → 151 (2.6×) |
| NCHW → NHWC | 64×256×56×56 | float32 | 11.1 → 11.2 (1.0×) | 70.3 → 63.1 (0.90×) | 199 → 197 (1.0×) |
| NHWC → NCHW | 64×56×56×256 | float32 | 3.7 → 9.2 (2.5×) | 24.1 → 59.6 (2.5×) | 56.9 → 195 (3.4×) |
| images, CHW → HWC | 64×3×224×224 | uint8 | 2.3 → 11.0 (4.7×) | 18.6 → 84.2 (4.5×) | 71.5 → 277 (3.9×) |
| images, HWC → CHW | 64×224×224×3 | uint8 | 3.9 → 7.6 (2.0×) | 30.9 → 58.2 (1.9×) | 115 → 182 (1.6×) |
| (B, S, H, D) → (B, H, S, D) | 32×2048×32×128 | bfloat16 | 22.8 → 19.5 (0.85×) | 80.7 → 81.9 (1.0×) | 203 → 235 (1.2×) |
| 8-D, (3, 4, 0, 1, 2, 5, 6, 7) | 12×32×8×16×24×16×20×3 | uint16 | 11.9 → 19.8 (1.7×) | 58.1 → 79.7 (1.4×) | 158 → 199 (1.3×) |
| 8-D, (6, 7, 3, 4, 5, 0, 1, 2) | 12×32×8×16×24×16×20×3 | uint16 | 0.39 → 4.6 (12×) | 2.8 → 35.2 (13×) | 9.8 → 117 (12×) |
| 8-D, (4, 5, 6, 7, 1, 2, 3, 0) | 12×32×8×16×24×16×20×3 | uint16 | 0.82 → 4.1 (5.0×) | 5.7 → 26.2 (4.6×) | 13.2 → 60.7 (4.6×) |
| 8-D, (6, 5, 4, 3, 2, 1, 0, 7) | 12×32×8×16×24×16×20×3 | uint16 | 0.43 → 4.8 (11×) | 3.2 → 33.4 (10×) | 12.5 → 99.0 (7.9×) |
| 8-D, (3, 4, 0, 1, 2, 5, 6, 7) | 12×32×8×16×24×16×20×3 | float32 | 21.3 → 24.6 (1.2×) | 73.3 → 80.9 (1.1×) | 197 → 213 (1.1×) |
| 8-D, (6, 7, 3, 4, 5, 0, 1, 2) | 12×32×8×16×24×16×20×3 | float32 | 0.71 → 4.3 (6.1×) | 5.5 → 24.7 (4.5×) | 19.1 → 93.2 (4.9×) |
| 8-D, (4, 5, 6, 7, 1, 2, 3, 0) | 12×32×8×16×24×16×20×3 | float32 | 1.4 → 5.4 (3.8×) | 7.3 → 36.3 (5.0×) | 23.2 → 82.4 (3.6×) |
| 8-D, (6, 5, 4, 3, 2, 1, 0, 7) | 12×32×8×16×24×16×20×3 | float32 | 0.79 → 6.4 (8.1×) | 6.1 → 43.3 (7.1×) | 22.5 → 125 (5.5×) |

Both sides write into an output allocated beforehand, so only the permutation is timed: `out.copy_(x.permute(p))` for
PyTorch and `hp.permute(x, p, out=out)` here. AMD EPYC 9334 (Zen 4, 2.7 GHz), Ubuntu 22.04, GCC 11.4 `-O2`; 1 and
8 threads on one 8-core CCD, 32 threads on one socket (`taskset`), the same thread count for both libraries; best
time of at least 5 calls and 0.2 s; best of two runs on two machines. The script is `bench/bench_torch.py`; the raw
outputs are in `bench/results/`.

- The gains are largest when the permutation moves the last (innermost) dimension in small pieces, which PyTorch's
  generic copy handles one element at a time: 3.6-13x for the 8-D permutations that move the last dimension, and
  1.6-5.8x for NHWC → NCHW, batched transposes and 8-bit images. A large 2-D transpose gains 1.5x on one thread and
  up to 29x on 32, because PyTorch runs it on one thread whatever the thread count.
- When whole rows stay together, as in (B, S, H, D) → (B, H, S, D) or the 8-D permutation (3, 4, 0, 1, 2, 5, 6, 7),
  both copy rows at memory speed: 0.85x to 1.7x. PyTorch is 15% faster on one thread for the 256-byte rows of the
  attention-head swap.
- PyTorch has a dedicated kernel for NCHW → NHWC in float32 (channels last); there the two are about equal, and this
  library is 10% slower on 8 threads.
- What most code writes, `x.permute(p).contiguous()`, also allocates a new tensor on every call and is slower than
  the PyTorch column above: 2.9 GB/s for the 2-D float32 transpose at every thread count (`bench_torch.py` prints it
  too).

## Install and use

### Python

```sh
pip install git+https://github.com/icodywu/High-Performance-Tensor-Permute
```

This compiles one C++ file, so it needs a C++17 compiler (GCC, Clang or MSVC). The package has no required
dependencies; it works with whichever of NumPy and PyTorch is installed.

```python
import numpy as np, torch
import hptpermute as hp

x = torch.randn(64, 256, 56, 56)
y = hp.permute(x, (0, 2, 3, 1))            # NCHW -> NHWC, a new contiguous tensor
hp.permute(x, (0, 2, 3, 1), out=y)         # into an existing contiguous tensor of the right shape and dtype
z = hp.contiguous(x.transpose(1, 3))       # a permuted view made contiguous, like .contiguous()
a = hp.permute(np.ones((3, 4, 5), np.float32), (2, 0, 1), num_threads=4)
```

- Any dtype works: elements are copied as opaque blocks of bytes (bool, bfloat16, complex, NumPy structured
  types, ...). Arrays of Python objects are refused.
- `x` can be any CPU array or tensor. A permuted view of a contiguous one (`x.T`, `x.transpose(1, 2)`,
  `x.permute(...)`) is read in place; other layouts (slices with steps, negative strides) are first copied to
  contiguous memory.
- `num_threads=None` (the default) picks the number of threads from the tensor size: one per 2 MiB, at most 16 and at
  most the number of hardware threads. `hp.auto_threads(nbytes)` tells what it picks. A given `num_threads` is an
  upper bound: no call uses more than one thread per 64 KiB of tensor.
- The call releases the GIL. It does not record autograd history: the result is a plain tensor.

### C++ (header only)

Copy `include/hptp.h` and `include/hptp.hpp`, or use CMake (below), and link with the threads library
(`-pthread`).

```cpp
#include "hptp.hpp"

std::vector<float> src(2 * 3 * 4), dst(src.size());
int rc = hptp::permute(src.data(), dst.data(), sizeof(float), {2, 3, 4}, {2, 0, 1});      // dst is 4 x 2 x 3

const int64_t shape[] = {2, 3, 4}, perm[] = {2, 0, 1};
rc = hptp::permute(src.data(), dst.data(), sizeof(float), 3, shape, perm, /*num_threads=*/8);
```

`hptp::permute(src, dst, elem_size, ndim, shape, perm, num_threads = 0)` returns `HPTP_OK` (0) or a negative
error code and writes nothing on error: a `perm` that is not a permutation, a negative size, more than 64
dimensions, a tensor larger than `SIZE_MAX` bytes, or overlapping `src` and `dst`. `num_threads <= 0` picks the
number of threads automatically.

### C

`include/hptp.h` declares the same function for C, compiled into `libhptp` from `src/hptp.cpp`:

```c
#include "hptp.h"

int rc = hptp_permute(src, dst, sizeof(float), 3, shape, perm, 0);
if (rc != HPTP_OK) fprintf(stderr, "%s\n", hptp_strerror(rc));
```

### CMake

```sh
cmake -S . -B build && cmake --build build && ctest --test-dir build
```

builds the library, the tests and `bench_permute`. In another project, `add_subdirectory()` this repository and link
`hptp::headers` (header only) or `hptp::hptp` (the C library); after `cmake --install`, `find_package(hptp)` gives
the same targets.

## How it works

1. **Dimensions of size 1 are dropped.**
2. **Source dimensions that stay next to each other, in the same order, are merged into one.** For example, shape
   `(8, 9, 12, 6, 4, 6, 9, 10)` with permutation `(2, 3, 4, 5, 0, 1, 6, 7)` becomes shape `(72, 1728, 90)` with
   permutation `(1, 0, 2)`. Both steps take O(ndim) time.
3. **A small last dimension that stays last becomes part of the element.** If the last dimension is not moved and
   holds at most 16 bytes, its values move together as one element: shape `(20, 8, 16, 12, 6)` of 1-byte values with
   permutation `(2, 1, 3, 0, 4)` becomes shape `(20, 8, 16, 12)` of 6-byte elements.
4. **What is left is one of two kinds.**
   - *The last dimension stays last.* Every row of it is copied as one block.
   - *The last dimension moves.* The permutation is then a set of generalized transposes: for each index of the
     other dimensions, the 2-D slab formed by the source's last dimension (its columns) and the dimension that
     becomes the destination's last (its rows) is transposed. A batch of rows is moved one column at a time, so that
     every write to the destination is a contiguous run of about 32 bytes, and the slab is walked in blocks sized
     for the cache and the TLB. 1- and 2-byte elements are transposed 8 x 8 and 4 x 4 at a time inside 64-bit
     words. Elements of 3, 5-7 and 9-15 bytes are moved with one wider load and store that spills into the next
     element of the batch, which the next move overwrites; the last element of a batch is moved exactly.
5. **Threads split the work into contiguous ranges** of (slab, block of rows) or (row, piece of row) units, so no two
   threads write the same bytes. The threads are kept in a pool between calls (created on first use; idle threads
   spin briefly, then sleep), so a call does not pay for starting threads.

Offsets are updated incrementally, with no division or multiplication per element.

## Permute and convert in one pass

`hptp::permute_convert<TS, TD>(src, dst, ndim, shape, perm, num_threads)` permutes and converts each value from `TS`
to `TD` in the same pass, which saves a full read and write of the tensor and the temporary buffer. Values that fit
are converted exactly; otherwise integers saturate to the range of `TD`, floating-point to integer conversion
saturates (NaN becomes 0), and a narrowing floating-point conversion gives plus or minus infinity beyond the range
of `TD` (NaN stays NaN). `bool` is not supported. This part is C++ only.

## Tests

- `tests/test_permute.cpp` compares every result with a direct element-by-element reference: hand-picked cases
  (scalars, empty tensors, size-1 dimensions, identity permutations, merges, folds, blocks larger than the cache
  blocks) for element sizes 1-17, 24 and 33 and 1, 3, 8 or automatic threads; 1,500 random permutations of up to 8
  dimensions; six caller threads at once; the error codes; the value conversions; and the 2024 interface. CMake
  builds it three times: as is, with tiny blocking sizes so that small tensors take every blocking path, and without
  the 64-bit-word kernels.
- `tests/test_capi.c` checks the C interface from C.
- `tests/test_python.py` (pytest) compares the Python package with NumPy and PyTorch.
- CI runs them on Linux (GCC and Clang), macOS (Apple silicon), Windows (MSVC) and 32-bit x86, and the C++ tests
  under AddressSanitizer, UndefinedBehaviorSanitizer and ThreadSanitizer.

## Changes from the 2024 version

The code was reorganized in October 2026 into the header `include/hptp.hpp`, the C interface `include/hptp.h`, and
the Python package. The 2024 functions are still there, with the same arguments, in namespace `hptp`:
`hptp::permute(src, dst, dtypeSize, src_dims, src_ndim, permute_idx, dst_dims, nThreads)` and
`hptp::permute_dtypeConv<T_S, T_D>(...)`. Behavior that changed:

- `dst` is now always written. Before, when nothing needed to move (an identity permutation, a 1-D tensor, or one
  that only moved dimensions of size 1), the function printed a message and returned without writing `dst`.
- Arguments are checked, and errors are returned as negative codes instead of crashing (for example `nThreads = 0`
  divided by zero) or overflowing fixed arrays (at most 32 dimensions were supported, unchecked).
- Elements are copied with `memcpy()`, so unaligned buffers and odd element sizes are no longer undefined behavior.
- `permute_dtypeConv` accepts any two arithmetic types except `bool`. When narrowing `double` to `float` it turned
  NaN into +infinity; NaN now stays NaN. Values out of range now saturate in every conversion, including the
  same-size ones that used to wrap (`int32_t` to `uint32_t`) or were undefined behavior (`float` to `int32_t`).
- It is faster. The inner loop keeps its row offsets in registers (it used to load and store a pointer for every
  element), large transposes are blocked for the cache and the TLB, 1- and 2-byte elements move in 8 x 8 and 4 x 4
  tiles, rows that stay together are copied without a `memcpy()` call when short, and threads come from a pool and
  share the work evenly even when the first dimension is small. Measured with `bench/bench_permute.cpp` on the same
  machine, it is 1.1x to 6.7x faster than the 2024 code on one thread (`bench/results/epyc9334_vs_2024_code.txt`).

## Limitations

- CPU only, and dense tensors only: the C and C++ functions take a row-major (C-order) source. The Python package
  reads permuted views in place but copies other layouts first.
- The blocking sizes were tuned on one CPU (AMD Zen 4); they are compile-time macros (`HPTP_*` in `hptp.hpp`) for
  other machines.
- Idle pool threads spin for a fraction of a millisecond before they sleep, as OpenMP threads do. In a process that
  also runs multi-threaded PyTorch operations, the two sets of threads can compete for cores right after each other's
  calls. For tensors of a few MB with 8 or more threads, the first calls after the threads are created can also take
  about twice as long, until the operating system has spread the threads over the cores (tens of milliseconds).
- 1- and 2-byte elements use the 64-bit-word kernels only on little-endian machines (all common ones).

## License

BSD 3-Clause, see [LICENSE](LICENSE). Copyright (c) 2024-2026 Yingquan (Cody) Wu.
