/*
 * hptp.h - High-Performance Tensor Permute, C interface.
 *
 * Copyright (c) 2024-2026, Yingquan (Cody) Wu. BSD-3-Clause license, see LICENSE.
 *
 * hptp_permute() writes the permutation of a dense, row-major (C-order) tensor into a separate dense,
 * row-major tensor: dst has shape (shape[perm[0]], ..., shape[perm[ndim-1]]) and
 *     dst[i_0, ..., i_{ndim-1}] = src[j_0, ..., j_{ndim-1}]  with  j_{perm[k]} = i_k,
 * the same result as numpy.ascontiguousarray(numpy.transpose(src, perm)) or src.permute(perm).contiguous() in
 * PyTorch. Elements are opaque blocks of elem_size bytes, so any data type works.
 *
 * The C++ header hptp.hpp has the same function as a header-only template library, plus permute_convert(),
 * which permutes and converts the element type in one pass.
 */
#ifndef HPTP_H
#define HPTP_H

#include <stddef.h>
#include <stdint.h>

#define HPTP_VERSION_MAJOR 1
#define HPTP_VERSION_MINOR 0
#define HPTP_VERSION_PATCH 0
#define HPTP_VERSION_STRING "1.0.0"

/* Largest supported number of dimensions. */
#define HPTP_MAX_DIMS 64

/* Return codes. Every check is made before anything is written to dst. */
#define HPTP_OK 0
#define HPTP_ERR_NULL (-1)    /* src, dst, shape or perm is NULL (src and dst may be NULL for an empty tensor) */
#define HPTP_ERR_NDIM (-2)    /* ndim < 0 or ndim > HPTP_MAX_DIMS */
#define HPTP_ERR_PERM (-3)    /* perm is not a permutation of 0, ..., ndim-1 */
#define HPTP_ERR_SHAPE (-4)   /* a negative size, or a tensor of more than SIZE_MAX bytes */
#define HPTP_ERR_ELEM (-5)    /* elem_size is 0 */
#define HPTP_ERR_OVERLAP (-6) /* src and dst overlap */

#if defined(_WIN32) && defined(HPTP_SHARED)
#  if defined(HPTP_BUILDING)
#    define HPTP_API __declspec(dllexport)
#  else
#    define HPTP_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(HPTP_BUILDING)
#  define HPTP_API __attribute__((visibility("default")))
#else
#  define HPTP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Permutes the ndim-dimensional tensor src of the given shape into dst. src and dst must not overlap.
 * num_threads > 0 uses that many threads (fewer if the tensor is small); num_threads <= 0 picks a number from the
 * tensor size and the number of hardware threads. Returns HPTP_OK or one of the negative HPTP_ERR_ codes.
 * ndim == 0 (a scalar) copies one element.
 */
HPTP_API int hptp_permute(const void* src, void* dst, size_t elem_size, int ndim, const int64_t* shape,
                          const int64_t* perm, int num_threads);

/* The number of threads that num_threads <= 0 would use for a tensor of the given size in bytes. */
HPTP_API int hptp_auto_threads(size_t bytes);

/* A short description of a return code. */
HPTP_API const char* hptp_strerror(int code);

/* The library version, "major.minor.patch". */
HPTP_API const char* hptp_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HPTP_H */
