/*
 * hptp.cpp - the C interface of hptp.h, compiled from the header-only implementation in hptp.hpp.
 *
 * Copyright (c) 2024-2026, Yingquan (Cody) Wu. BSD-3-Clause license, see LICENSE.
 */
#include "hptp.hpp"

extern "C" {

HPTP_API int hptp_permute(const void* src, void* dst, size_t elem_size, int ndim, const int64_t* shape,
                          const int64_t* perm, int num_threads) {
    return hptp::permute(src, dst, elem_size, ndim, shape, perm, num_threads);
}

HPTP_API int hptp_auto_threads(size_t bytes) { return hptp::detail::auto_threads(bytes); }

HPTP_API const char* hptp_strerror(int code) {
    switch (code) {
    case HPTP_OK: return "success";
    case HPTP_ERR_NULL: return "a required pointer is NULL";
    case HPTP_ERR_NDIM: return "ndim is negative or larger than HPTP_MAX_DIMS";
    case HPTP_ERR_PERM: return "perm is not a permutation of 0..ndim-1";
    case HPTP_ERR_SHAPE: return "a negative size, or the tensor is larger than SIZE_MAX bytes";
    case HPTP_ERR_ELEM: return "elem_size is 0";
    case HPTP_ERR_OVERLAP: return "src and dst overlap";
    default: return "unknown error";
    }
}

HPTP_API const char* hptp_version(void) { return HPTP_VERSION_STRING; }

#ifdef HPTP_PYTHON_EXTENSION
/* The Python package builds this file as an extension module so that pip compiles it; the module is then loaded
 * with ctypes, never imported, and this entry point only satisfies the build. */
HPTP_API void* PyInit__hptp(void) { return 0; }
#endif

}  // extern "C"
