/*
 * hptp.hpp - High-Performance Tensor Permute, header-only C++17 implementation.
 *
 * Copyright (c) 2024-2026, Yingquan (Cody) Wu. BSD-3-Clause license, see LICENSE.
 *
 *   int hptp::permute(const void* src, void* dst, size_t elem_size, int ndim, const int64_t* shape,
 *                     const int64_t* perm, int num_threads = 0);
 *       Permutes a dense row-major tensor into a separate dense row-major tensor (the exact definition is in
 *       hptp.h). Returns HPTP_OK (0) or a negative HPTP_ERR_ code.
 *
 *   template <class TS, class TD>
 *   int hptp::permute_convert(const TS* src, TD* dst, int ndim, const int64_t* shape, const int64_t* perm,
 *                             int num_threads = 0);
 *       The same permutation, converting each value from TS to TD in the same pass (see hptp::convert()).
 *
 * The 2024 interface, permute(src, dst, dtypeSize, src_dims, src_ndim, permute_idx, dst_dims, nThreads) and
 * permute_dtypeConv<T_S, T_D>(...), is kept as overloads in namespace hptp.
 *
 * How it works:
 * 1. Dimensions of size 1 are dropped.
 * 2. Source dimensions that stay adjacent and in the same order are merged:
 *        shape (8, 9, 12, 6, 4, 6, 9, 10), perm (2, 3, 4, 5, 0, 1, 6, 7)  ->  shape (72, 1728, 90), perm (1, 0, 2).
 * 3. If the last dimension stays last and holds at most 16 bytes, it is folded into the element:
 *        shape (20, 8, 16, 12, 6) of 1-byte values, perm (2, 1, 3, 0, 4)  ->  shape (20, 8, 16, 12) of 6-byte elements.
 * 4. If the last dimension still stays last ("type B"), each row of it is copied as one block. Otherwise
 *    ("type A") the permutation is a set of generalized transposes: for every index of the other dimensions, the
 *    2-D slab spanned by the source's last dimension (columns) and the dimension that becomes the destination's last
 *    (rows) is transposed. A batch of rows is moved one column at a time, so that each write to the destination is
 *    a contiguous run of about 32 bytes, and the slab is processed in blocks sized for the cache and the TLB. 1- and
 *    2-byte elements are transposed 8 x 8 and 4 x 4 at a time inside 64-bit words. Elements of 3, 5-7 and 9-15
 *    bytes move with one wider load and store that spills into the next element of the batch, which the next move
 *    overwrites; the last element of a batch moves exactly.
 * 5. Threads take contiguous ranges of (slab, block of rows) or (row, piece of row) units, so their writes never
 *    overlap.
 * Offsets are updated incrementally, with no division or multiplication per element.
 */
#ifndef HPTP_HPP
#define HPTP_HPP

#include "hptp.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>
#if !defined(_WIN32)
#include <unistd.h>
#endif
#if defined(_MSC_VER)
#include <intrin.h>
#endif

/* Tuning (see blocking() below). A transpose block covers HPTP_ROW_BLOCK_BYTES of each destination column and at most
 * HPTP_BLOCK_BYTES of source. Source rows longer than HPTP_SHORT_ROW_BYTES are split into tiles of HPTP_WIDE_TILE_COLS
 * columns if destination columns are at least HPTP_WIDE_STRIDE_BYTES apart, else of HPTP_COL_TILE_BYTES. Automatic
 * threading uses one thread per HPTP_BYTES_PER_THREAD of tensor, up to HPTP_MAX_AUTO_THREADS and the number of hardware
 * threads; no call uses more than one thread per HPTP_MIN_BYTES_PER_THREAD. Idle pool threads spin for
 * HPTP_SPIN_ROUNDS pauses before they sleep. The defaults were tuned on AMD Zen 4 (32 KiB L1D, 1 MiB L2 per core). */
#ifndef HPTP_ROW_BLOCK_BYTES
#define HPTP_ROW_BLOCK_BYTES 4096
#endif
#ifndef HPTP_BLOCK_BYTES
#define HPTP_BLOCK_BYTES (512u << 10)
#endif
#ifndef HPTP_SHORT_ROW_BYTES
#define HPTP_SHORT_ROW_BYTES 4096
#endif
#ifndef HPTP_WIDE_STRIDE_BYTES
#define HPTP_WIDE_STRIDE_BYTES 8192
#endif
#ifndef HPTP_WIDE_TILE_COLS
#define HPTP_WIDE_TILE_COLS 128
#endif
#ifndef HPTP_COL_TILE_BYTES
#define HPTP_COL_TILE_BYTES 1024
#endif
#ifndef HPTP_BYTES_PER_THREAD
#define HPTP_BYTES_PER_THREAD (2u << 20)
#endif
#ifndef HPTP_MAX_AUTO_THREADS
#define HPTP_MAX_AUTO_THREADS 16
#endif
#ifndef HPTP_MIN_BYTES_PER_THREAD
#define HPTP_MIN_BYTES_PER_THREAD (64u << 10)
#endif
#ifndef HPTP_SPIN_ROUNDS
#define HPTP_SPIN_ROUNDS 20000
#endif

/* 1- and 2-byte elements are transposed 8 x 8 and 4 x 4 at a time inside 64-bit words on little-endian machines,
 * unless HPTP_NO_SWAR is defined. */
#if defined(HPTP_NO_SWAR)
#define HPTP_SWAR 0
#elif (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__) || defined(_WIN32)
#define HPTP_SWAR 1
#else
#define HPTP_SWAR 0
#endif

namespace hptp {

/* Value conversion of permute_convert(): exact when the value fits in TD. Otherwise integers saturate to TD's range,
 * floating-point values convert to an integer type with saturation and NaN -> 0, and a narrower floating-point type
 * gets +-infinity for values beyond its range (NaN stays NaN). bool is not supported. */
template <class TS, class TD>
inline TD convert(TS v) {
    static_assert(std::is_arithmetic<TS>::value && std::is_arithmetic<TD>::value, "arithmetic types only");
    static_assert(!std::is_same<TS, bool>::value && !std::is_same<TD, bool>::value, "bool is not supported");
    using LD = std::numeric_limits<TD>;
    if constexpr (std::is_same<TS, TD>::value) {
        return v;
    } else if constexpr (std::is_integral<TS>::value && std::is_integral<TD>::value) {
        if constexpr (std::is_signed<TS>::value) {
            const intmax_t x = v;
            if constexpr (std::is_signed<TD>::value) {
                if (x < static_cast<intmax_t>(LD::min())) return LD::min();
                if (x > static_cast<intmax_t>(LD::max())) return LD::max();
            } else {
                if (x < 0) return 0;
                if (static_cast<uintmax_t>(x) > static_cast<uintmax_t>(LD::max())) return LD::max();
            }
        } else {
            const uintmax_t x = v;
            if (x > static_cast<uintmax_t>(LD::max())) return LD::max();
        }
        return static_cast<TD>(v);
    } else if constexpr (std::is_floating_point<TS>::value && std::is_integral<TD>::value) {
        if (v != v) return 0;
        if (v <= static_cast<TS>(LD::min())) return LD::min();
        if (v >= static_cast<TS>(LD::max())) return LD::max();  // TS(max) may round up; anything below it fits
        return static_cast<TD>(v);
    } else if constexpr (std::is_floating_point<TS>::value && std::is_floating_point<TD>::value) {
        if constexpr (sizeof(TD) < sizeof(TS)) {
            if (v > static_cast<TS>(LD::max())) return LD::infinity();
            if (v < -static_cast<TS>(LD::max())) return -LD::infinity();
        }
        return static_cast<TD>(v);
    } else {
        return static_cast<TD>(v);  // integer to floating point: rounds to nearest
    }
}

namespace detail {

constexpr int kMaxDims = HPTP_MAX_DIMS;
constexpr uint64_t kFoldBytes = 16;

// The permutation after steps 1-3.
struct Plan {
    int ndim = 0;                  // 0 or 1: a plain copy
    uint64_t dims[kMaxDims] = {};  // source shape, in elements of `group` values
    int perm[kMaxDims] = {};       // destination dimension i is source dimension perm[i]
    uint64_t group = 1;            // values per element (a folded last dimension)
    uint64_t numel = 0;            // values in the tensor
};

// Checks the arguments other than src and dst and builds the plan. value_size is the largest size of a source or
// destination value (for the size check); fold_size is the destination value size (for step 3).
inline int make_plan(int ndim, const int64_t* shape, const int64_t* perm, size_t value_size, size_t fold_size,
                     Plan& p) {
    if (ndim < 0 || ndim > kMaxDims) return HPTP_ERR_NDIM;
    if (ndim > 0 && (shape == nullptr || perm == nullptr)) return HPTP_ERR_NULL;
    if (value_size == 0) return HPTP_ERR_ELEM;
    bool seen[kMaxDims] = {};
    bool empty = false;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] < 0) return HPTP_ERR_SHAPE;
        if (perm[i] < 0 || perm[i] >= ndim || seen[perm[i]]) return HPTP_ERR_PERM;
        seen[perm[i]] = true;
        empty = empty || shape[i] == 0;
    }
    uint64_t numel = empty ? 0 : 1;
    const uint64_t limit = static_cast<uint64_t>(SIZE_MAX) / value_size;
    for (int i = 0; i < ndim && !empty; i++) {
        if (static_cast<uint64_t>(shape[i]) > limit / numel) return HPTP_ERR_SHAPE;
        numel *= static_cast<uint64_t>(shape[i]);
    }
    p.numel = numel;
    p.ndim = 0;
    p.group = 1;
    if (numel <= 1) return HPTP_OK;

    // 1. Drop dimensions of size 1.
    int newidx[kMaxDims];
    uint64_t d1[kMaxDims];
    int n = 0;
    for (int i = 0; i < ndim; i++) {
        newidx[i] = shape[i] > 1 ? n : -1;
        if (shape[i] > 1) d1[n++] = static_cast<uint64_t>(shape[i]);
    }
    int p1[kMaxDims];
    int m = 0;
    for (int i = 0; i < ndim; i++)
        if (newidx[perm[i]] >= 0) p1[m++] = newidx[perm[i]];

    // 2. Merge runs of source dimensions that stay adjacent and in order. Run r, in destination order, is the
    //    source dimensions runStart[r] .. runEnd[r]-1.
    int runStart[kMaxDims], runEnd[kMaxDims], nr = 0;
    for (int i = 0; i < n;) {
        int j = i + 1;
        while (j < n && p1[j] == p1[j - 1] + 1) j++;
        runStart[nr] = p1[i];
        runEnd[nr] = p1[i] + (j - i);
        nr++;
        i = j;
    }
    int runAt[kMaxDims];  // the run that starts at source dimension s, or -1
    for (int s = 0; s < n; s++) runAt[s] = -1;
    for (int r = 0; r < nr; r++) runAt[runStart[r]] = r;
    int rank[kMaxDims];
    for (int s = 0, k = 0; s < n; s++) {
        const int r = runAt[s];
        if (r < 0) continue;
        uint64_t size = 1;
        for (int t = runStart[r]; t < runEnd[r]; t++) size *= d1[t];
        p.dims[k] = size;
        rank[r] = k++;
    }
    for (int r = 0; r < nr; r++) p.perm[r] = rank[r];
    p.ndim = nr;

    // 3. Fold a small last dimension that stays last into the element.
    if (nr >= 2 && p.perm[nr - 1] == nr - 1 && p.dims[nr - 1] * fold_size <= kFoldBytes) {
        p.group = p.dims[nr - 1];
        p.ndim = nr - 1;
    }
    return HPTP_OK;
}

// Element strides of each source dimension k: sw[k] in the source, dw[k] in the destination.
inline void strides(const Plan& p, uint64_t* sw, uint64_t* dw) {
    const int n = p.ndim;
    sw[n - 1] = 1;
    for (int k = n - 1; k > 0; k--) sw[k - 1] = sw[k] * p.dims[k];
    uint64_t w = 1;
    for (int i = n - 1; i >= 0; i--) {
        dw[p.perm[i]] = w;
        w *= p.dims[p.perm[i]];
    }
}

// Dimensions iterated as an odometer, outermost first, with their source and destination strides.
struct Outer {
    int n = 0;
    uint64_t size[kMaxDims], sw[kMaxDims], dw[kMaxDims];
    void add(uint64_t sz, uint64_t s, uint64_t d) {
        size[n] = sz;
        sw[n] = s;
        dw[n] = d;
        n++;
    }
    uint64_t count() const {
        uint64_t c = 1;
        for (int k = 0; k < n; k++) c *= size[k];
        return c;
    }
};

// Calls f(src_offset, dst_offset, index) for the odometer indices first .. last-1.
template <class F>
inline void for_outer(const Outer& o, uint64_t first, uint64_t last, F&& f) {
    uint64_t idx[kMaxDims];
    uint64_t s = 0, d = 0, rem = first;
    for (int k = o.n - 1; k >= 0; k--) {
        idx[k] = rem % o.size[k];
        rem /= o.size[k];
        s += idx[k] * o.sw[k];
        d += idx[k] * o.dw[k];
    }
    for (uint64_t i = first; i < last; i++) {
        f(s, d, i);
        for (int k = o.n - 1; k >= 0; k--) {
            if (++idx[k] < o.size[k]) {
                s += o.sw[k];
                d += o.dw[k];
                break;
            }
            idx[k] = 0;
            s -= (o.size[k] - 1) * o.sw[k];
            d -= (o.size[k] - 1) * o.dw[k];
        }
    }
}

// Start of part k when total is split into `parts` nearly equal contiguous parts.
inline uint64_t split(uint64_t total, uint64_t parts, uint64_t k) {
    return k * (total / parts) + std::min(k, total % parts);
}

inline int auto_threads(uint64_t bytes) {
    const unsigned hw = std::thread::hardware_concurrency();
    const uint64_t cap = std::min<uint64_t>(hw ? hw : 1, HPTP_MAX_AUTO_THREADS);
    return static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(bytes / HPTP_BYTES_PER_THREAD, cap)));
}

// The number of threads to use: as requested (at most 1024) or automatic, and at most one per
// HPTP_MIN_BYTES_PER_THREAD of tensor.
inline int resolve_threads(int requested, uint64_t bytes) {
    const int n = requested > 0 ? std::min(requested, 1024) : auto_threads(bytes);
    return static_cast<int>(std::min<uint64_t>(n, std::max<uint64_t>(1, bytes / HPTP_MIN_BYTES_PER_THREAD)));
}

// Lets other hyperthreads run while spinning.
inline void cpu_relax() {
#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    _mm_pause();
#elif defined(_MSC_VER) && defined(_M_ARM64)
    __yield();
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
    __builtin_ia32_pause();
#elif defined(__GNUC__) && defined(__aarch64__)
    __asm__ __volatile__("yield" ::: "memory");
#endif
}

// One step of a spin-wait: mostly a pause, sometimes a yield, so that a spinning thread gives its core to a thread
// that has work (when there are more threads than cores).
inline void pause_or_yield(int i) {
    if ((i & 15) == 15)
        std::this_thread::yield();
    else
        cpu_relax();
}

// Worker threads kept between calls, so that a call does not pay for creating threads. run(n, call, ctx) runs
// call(ctx, t) for t = 0 .. n-1, t = 0 in the calling thread, and returns when all are done; it returns false, having
// run nothing, if another call is using the pool. Each worker has its own wake-up slot and, after a job, spins for a
// short while (HPTP_SPIN_ROUNDS pauses) before it sleeps, so that back-to-back calls find it awake and on its core.
// The pool is created on first use, grows to the largest n, and is never destroyed (its threads wait until the
// process exits). A child process made by fork() gets a new pool.
class ThreadPool {
  public:
    using Call = void (*)(void*, int);

    static ThreadPool& get() {
        static std::atomic<ThreadPool*> current{nullptr};
        ThreadPool* p = current.load(std::memory_order_acquire);
        if (p != nullptr && p->pid_ == process_id()) return *p;
        ThreadPool* fresh = new ThreadPool();
        if (current.compare_exchange_strong(p, fresh, std::memory_order_acq_rel)) return *fresh;
        delete fresh;  // another thread installed one first
        return *p;
    }

    bool run(int n, Call call, void* ctx) {
        if (!busy_.try_lock()) return false;
        try {
            while (static_cast<int>(slots_.size()) < n - 1) {
                slots_.emplace_back(new Slot());
                Slot* slot = slots_.back().get();
                const int id = static_cast<int>(slots_.size());
                slot->thread = std::thread([this, slot, id] { work(slot, id); });
            }
        } catch (...) {
            if (!slots_.empty() && !slots_.back()->thread.joinable()) slots_.pop_back();
            n = static_cast<int>(slots_.size()) + 1;  // use the threads there are
        }
        call_ = call;
        ctx_ = ctx;
        generation_++;
        remaining_.store(n - 1, std::memory_order_relaxed);
        for (int i = 0; i < n - 1; i++) {
            Slot& slot = *slots_[static_cast<size_t>(i)];
            {
                std::lock_guard<std::mutex> lock(slot.m);
                slot.go.store(generation_, std::memory_order_release);
            }
            slot.cv.notify_one();
        }
        call(ctx, 0);
        for (int i = 0; i < HPTP_SPIN_ROUNDS && remaining_.load(std::memory_order_acquire) != 0; i++)
            pause_or_yield(i);
        if (remaining_.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(done_m_);
            done_cv_.wait(lock, [this] { return remaining_.load(std::memory_order_acquire) == 0; });
        }
        busy_.unlock();
        return true;
    }

  private:
    struct Slot {
        std::mutex m;
        std::condition_variable cv;
        std::atomic<uint64_t> go{0};  // the generation to run
        std::thread thread;
    };

    static long process_id() {
#if defined(_WIN32)
        return 0;
#else
        return static_cast<long>(::getpid());
#endif
    }

    void work(Slot* slot, int id) {
        uint64_t seen = 0;
        for (;;) {
            uint64_t g = slot->go.load(std::memory_order_acquire);
            for (int i = 0; g == seen && i < HPTP_SPIN_ROUNDS; i++) {
                pause_or_yield(i);
                g = slot->go.load(std::memory_order_acquire);
            }
            if (g == seen) {
                std::unique_lock<std::mutex> lock(slot->m);
                slot->cv.wait(lock, [&] { return (g = slot->go.load(std::memory_order_acquire)) != seen; });
            }
            seen = g;
            call_(ctx_, id);
            if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                std::lock_guard<std::mutex> lock(done_m_);
                done_cv_.notify_one();
            }
        }
    }

    const long pid_ = process_id();
    std::mutex busy_;
    std::vector<std::unique_ptr<Slot>> slots_;
    Call call_ = nullptr;
    void* ctx_ = nullptr;
    uint64_t generation_ = 0;
    std::atomic<int> remaining_{0};
    std::mutex done_m_;
    std::condition_variable done_cv_;
};

// Runs fn(first, last) on up to `threads` contiguous ranges that split 0 .. units-1, on the thread pool; the calling
// thread takes the first range. If the pool is busy (another call in progress), new threads are started instead, and
// a range whose thread cannot be started runs in the calling thread.
template <class Fn>
inline void parallel_for(uint64_t units, int threads, Fn&& fn) {
    if (units == 0) return;
    const uint64_t nt = std::min<uint64_t>(threads > 1 ? static_cast<uint64_t>(threads) : 1, units);
    if (nt == 1) {
        fn(uint64_t(0), units);
        return;
    }
    struct Job {
        Fn* fn;
        uint64_t units, nt;
    } job{&fn, units, nt};
    const auto call = [](void* ctx, int t) {
        const Job& j = *static_cast<Job*>(ctx);
        (*j.fn)(split(j.units, j.nt, static_cast<uint64_t>(t)), split(j.units, j.nt, static_cast<uint64_t>(t) + 1));
    };
    if (ThreadPool::get().run(static_cast<int>(nt), call, &job)) return;
    std::vector<std::thread> extra;
    uint64_t t = 1;
    try {
        extra.reserve(nt - 1);
        for (; t < nt; t++) {
            const uint64_t a = split(units, nt, t), b = split(units, nt, t + 1);
            extra.emplace_back([&fn, a, b] { fn(a, b); });
        }
    } catch (...) {
    }
    fn(split(units, nt, 0), split(units, nt, 1));
    for (uint64_t u = t; u < nt; u++) fn(split(units, nt, u), split(units, nt, u + 1));
    for (auto& th : extra) th.join();
}

inline bool overlaps(const void* a, uint64_t na, const void* b, uint64_t nb) {
    const auto x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    return x < y + nb && y < x + na;
}

// Exchanges the off-diagonal blocks of two 64-bit rows: the parts of a selected by ~mask and of b selected by mask.
inline void swap_blocks(uint64_t& a, uint64_t& b, uint64_t mask, unsigned shift) {
    const uint64_t x = a, y = b;
    a = (x & mask) | ((y & mask) << shift);
    b = ((x >> shift) & mask) | (y & ~mask);
}

// Transposes 8 x 8 bytes: 8 bytes from each of 8 source rows (srow bytes apart) to 8 bytes in each of 8 destination
// columns (dcol bytes apart). Little-endian only.
inline void transpose8x8_u8(uint8_t* d, uint64_t dcol, const uint8_t* s, uint64_t srow) {
    uint64_t w0, w1, w2, w3, w4, w5, w6, w7;
    std::memcpy(&w0, s, 8);
    std::memcpy(&w1, s + srow, 8);
    std::memcpy(&w2, s + 2 * srow, 8);
    std::memcpy(&w3, s + 3 * srow, 8);
    std::memcpy(&w4, s + 4 * srow, 8);
    std::memcpy(&w5, s + 5 * srow, 8);
    std::memcpy(&w6, s + 6 * srow, 8);
    std::memcpy(&w7, s + 7 * srow, 8);
    const uint64_t m4 = 0x00000000FFFFFFFFull, m2 = 0x0000FFFF0000FFFFull, m1 = 0x00FF00FF00FF00FFull;
    swap_blocks(w0, w4, m4, 32);
    swap_blocks(w1, w5, m4, 32);
    swap_blocks(w2, w6, m4, 32);
    swap_blocks(w3, w7, m4, 32);
    swap_blocks(w0, w2, m2, 16);
    swap_blocks(w1, w3, m2, 16);
    swap_blocks(w4, w6, m2, 16);
    swap_blocks(w5, w7, m2, 16);
    swap_blocks(w0, w1, m1, 8);
    swap_blocks(w2, w3, m1, 8);
    swap_blocks(w4, w5, m1, 8);
    swap_blocks(w6, w7, m1, 8);
    std::memcpy(d, &w0, 8);
    std::memcpy(d + dcol, &w1, 8);
    std::memcpy(d + 2 * dcol, &w2, 8);
    std::memcpy(d + 3 * dcol, &w3, 8);
    std::memcpy(d + 4 * dcol, &w4, 8);
    std::memcpy(d + 5 * dcol, &w5, 8);
    std::memcpy(d + 6 * dcol, &w6, 8);
    std::memcpy(d + 7 * dcol, &w7, 8);
}

// Transposes 4 x 4 two-byte elements, as above.
inline void transpose4x4_u16(uint8_t* d, uint64_t dcol, const uint8_t* s, uint64_t srow) {
    uint64_t w0, w1, w2, w3;
    std::memcpy(&w0, s, 8);
    std::memcpy(&w1, s + srow, 8);
    std::memcpy(&w2, s + 2 * srow, 8);
    std::memcpy(&w3, s + 3 * srow, 8);
    swap_blocks(w0, w2, 0x00000000FFFFFFFFull, 32);
    swap_blocks(w1, w3, 0x00000000FFFFFFFFull, 32);
    swap_blocks(w0, w1, 0x0000FFFF0000FFFFull, 16);
    swap_blocks(w2, w3, 0x0000FFFF0000FFFFull, 16);
    std::memcpy(d, &w0, 8);
    std::memcpy(d + dcol, &w1, 8);
    std::memcpy(d + 2 * dcol, &w2, 8);
    std::memcpy(d + 3 * dcol, &w3, 8);
}

// Element movers. fast() may read and write up to W bytes, spilling into the next element; exact() moves one element.
// kTile > 0: tile(d, dcol, s, srow) transposes kTile x kTile elements at once.
template <size_t N>
struct CopyN {
    static constexpr size_t W = N == 3 ? 4 : (N >= 5 && N <= 7) ? 8 : (N >= 9 && N <= 15) ? 16 : N;
    static constexpr int kTile = !HPTP_SWAR ? 0 : N == 1 ? 8 : N == 2 ? 4 : 0;
    size_t src_size() const { return N; }
    size_t dst_size() const { return N; }
    void fast(uint8_t* d, const uint8_t* s) const { std::memcpy(d, s, W); }
    void exact(uint8_t* d, const uint8_t* s) const { std::memcpy(d, s, N); }
    static void tile(uint8_t* d, uint64_t dcol, const uint8_t* s, uint64_t srow) {
        if constexpr (N == 1)
            transpose8x8_u8(d, dcol, s, srow);
        else
            transpose4x4_u16(d, dcol, s, srow);
    }
};

template <class M, class = void>
struct tile_of {
    static constexpr int value = 0;
};
template <class M>
struct tile_of<M, std::void_t<decltype(M::kTile)>> {
    static constexpr int value = M::kTile;
};

struct CopyAny {
    size_t n;
    size_t src_size() const { return n; }
    size_t dst_size() const { return n; }
    void fast(uint8_t* d, const uint8_t* s) const { std::memcpy(d, s, n); }
    void exact(uint8_t* d, const uint8_t* s) const { std::memcpy(d, s, n); }
};

template <class TS, class TD>
struct ConvertOne {
    size_t src_size() const { return sizeof(TS); }
    size_t dst_size() const { return sizeof(TD); }
    void fast(uint8_t* d, const uint8_t* s) const { exact(d, s); }
    void exact(uint8_t* d, const uint8_t* s) const {
        *reinterpret_cast<TD*>(d) = convert<TS, TD>(*reinterpret_cast<const TS*>(s));
    }
};

template <class TS, class TD>
struct ConvertGroup {
    uint64_t g;
    size_t src_size() const { return g * sizeof(TS); }
    size_t dst_size() const { return g * sizeof(TD); }
    void fast(uint8_t* d, const uint8_t* s) const { exact(d, s); }
    void exact(uint8_t* d, const uint8_t* s) const {
        const TS* sp = reinterpret_cast<const TS*>(s);
        TD* dp = reinterpret_cast<TD*>(d);
        for (uint64_t k = 0; k < g; k++) dp[k] = convert<TS, TD>(sp[k]);
    }
};

// Rows per batch: about 32 bytes of destination per column.
constexpr int batch_rows(size_t esize) {
    return esize == 1   ? 32
           : esize == 2 ? 16
           : esize == 3 ? 12
           : esize == 4 ? 8
           : esize == 5 ? 7
           : esize == 6 ? 6
           : esize == 7 ? 5
           : esize == 8 ? 4
           : esize <= 16 ? 3
           : esize <= 32 ? 2
                         : 1;
}

// A slab of a type-A permutation: element (r, c) is src[r * srcWt + c] and dst[c * dstWt + r].
struct Slab {
    uint64_t rows, cols, srcWt, dstWt;
};

// Block of a slab moved at a time: `rows` rows (a multiple of the batch) by `cols` columns. Short source rows are not
// split; long ones are split into tiles of HPTP_WIDE_TILE_COLS columns when every destination column lies on its own
// pages (few pages per pass), else of HPTP_COL_TILE_BYTES. Rows per block are limited so that the block's source fits
// in HPTP_BLOCK_BYTES.
struct Blocking {
    uint64_t rows, cols;
};

template <int B>
inline Blocking blocking(const Slab& sl, size_t ss, size_t ds) {
    Blocking b;
    if (sl.cols * ss <= HPTP_SHORT_ROW_BYTES)
        b.cols = sl.cols;
    else if (sl.dstWt * ds >= HPTP_WIDE_STRIDE_BYTES)
        b.cols = HPTP_WIDE_TILE_COLS;
    else
        b.cols = std::max<uint64_t>(1, HPTP_COL_TILE_BYTES / ss);
    const uint64_t source_row = std::min(b.cols, sl.cols) * ss;
    const uint64_t rows = std::min<uint64_t>(HPTP_ROW_BLOCK_BYTES / ds, HPTP_BLOCK_BYTES / source_row);
    b.rows = std::max<uint64_t>(B, rows / B * B);
    return b;
}

// One column of a batch: element J of the batch comes from source row J (srow bytes apart) and goes to the J-th
// destination element. All but the last element use the fast (spilling) move.
template <int B, int J, class M>
inline void move_one(uint8_t* d, const uint8_t* s, uint64_t srow, const M& mv) {
    if constexpr (J + 1 < B)
        mv.fast(d + J * mv.dst_size(), s + J * srow);
    else
        mv.exact(d + J * mv.dst_size(), s + J * srow);
}

template <int B, class M, int... J>
inline void move_batch(uint8_t* d, const uint8_t* s, uint64_t srow, const M& mv, std::integer_sequence<int, J...>) {
    (move_one<B, J>(d, s, srow, mv), ...);
}

// R rows (R < B, the rest of a block) for nc columns, unrolled for this R.
template <int R, class M>
inline void move_columns(uint8_t* d, const uint8_t* s, uint64_t nc, uint64_t srow, uint64_t dcol, const M& mv) {
    const size_t ss = mv.src_size();
    for (uint64_t c = nc; c > 0; c--) {
        move_batch<R>(d, s, srow, mv, std::make_integer_sequence<int, R>());
        s += ss;
        d += dcol;
    }
}

template <class M, int... R>
inline void move_columns_rem([[maybe_unused]] int rem, [[maybe_unused]] uint8_t* d, [[maybe_unused]] const uint8_t* s,
                             [[maybe_unused]] uint64_t nc, [[maybe_unused]] uint64_t srow,
                             [[maybe_unused]] uint64_t dcol, [[maybe_unused]] const M& mv,
                             std::integer_sequence<int, R...>) {
    (void)((rem == R + 1 && (move_columns<R + 1>(d, s, nc, srow, dcol, mv), true)) || ...);
}

// Moves rows r0 .. r1-1 of a slab, block by block, B rows at a time.
template <int B, class M>
inline void transpose_rows(const uint8_t* src, uint8_t* dst, const Slab& sl, const Blocking& bl, uint64_t r0,
                           uint64_t r1, const M& mv) {
    const size_t ss = mv.src_size(), ds = mv.dst_size();
    const uint64_t srow = sl.srcWt * ss, dcol = sl.dstWt * ds;
    for (uint64_t rbeg = r0; rbeg < r1; rbeg += bl.rows) {
        const uint64_t rend = std::min(rbeg + bl.rows, r1);
        for (uint64_t c0 = 0; c0 < sl.cols; c0 += bl.cols) {
            const uint64_t nc = std::min(bl.cols, sl.cols - c0);
            uint64_t r = rbeg;
            for (; r + B <= rend; r += B) {
                const uint8_t* s = src + (r * sl.srcWt + c0) * ss;
                uint8_t* d = dst + (c0 * sl.dstWt + r) * ds;
                uint64_t c = nc;
                if constexpr (tile_of<M>::value > 0) {  // T columns at a time, as B/T tiles of T x T
                    constexpr int T = tile_of<M>::value;
                    static_assert(B % T == 0, "a batch is a whole number of tiles");
                    for (; c >= T; c -= T) {
                        for (int g = 0; g < B; g += T) M::tile(d + g * ds, dcol, s + g * srow, srow);
                        s += T * ss;
                        d += T * dcol;
                    }
                }
                for (; c > 0; c--) {
                    move_batch<B>(d, s, srow, mv, std::make_integer_sequence<int, B>());
                    s += ss;
                    d += dcol;
                }
            }
            if (r < rend)  // fewer than B rows left
                move_columns_rem(static_cast<int>(rend - r), dst + (c0 * sl.dstWt + r) * ds,
                                 src + (r * sl.srcWt + c0) * ss, nc, srow, dcol, mv,
                                 std::make_integer_sequence<int, B - 1>());
        }
    }
}

// Type A: the last dimension moves.
template <int B, class M>
inline void run_type_a(const uint8_t* src, uint8_t* dst, const Plan& p, const M& mv, int threads) {
    const int n = p.ndim;
    uint64_t sw[kMaxDims] = {}, dw[kMaxDims] = {};
    strides(p, sw, dw);
    const int R = p.perm[n - 1];
    const Slab sl{p.dims[R], p.dims[n - 1], sw[R], dw[n - 1]};
    Outer o;
    for (int k = 0; k < n - 1; k++)
        if (k != R) o.add(p.dims[k], sw[k], dw[k]);
    const uint64_t slabs = o.count();
    const size_t ss = mv.src_size(), ds = mv.dst_size();
    const Blocking bl = blocking<B>(sl, ss, ds);
    // Units of work: (slab, chunk of rows). A chunk is a row block, or one batch when that gives too few units.
    uint64_t chunk = bl.rows;
    uint64_t per = (sl.rows + chunk - 1) / chunk;
    if (threads > 1 && slabs * per < 4 * static_cast<uint64_t>(threads)) {
        chunk = B;
        per = (sl.rows + chunk - 1) / chunk;
    }
    parallel_for(slabs * per, threads, [&](uint64_t u0, uint64_t u1) {
        const uint64_t first = u0 / per, last = (u1 - 1) / per + 1;
        for_outer(o, first, last, [&](uint64_t s, uint64_t d, uint64_t i) {
            const uint64_t b0 = i == first ? u0 % per : 0;
            const uint64_t b1 = i == last - 1 ? (u1 - 1) % per + 1 : per;
            transpose_rows<B>(src + s * ss, dst + d * ds, sl, bl, b0 * chunk, std::min(b1 * chunk, sl.rows), mv);
        });
    });
}

template <class M>
inline void run_type_a_any(const uint8_t* src, uint8_t* dst, const Plan& p, const M& mv, int threads) {
    switch (batch_rows(mv.dst_size())) {
    case 32: return run_type_a<32>(src, dst, p, mv, threads);
    case 16: return run_type_a<16>(src, dst, p, mv, threads);
    case 12: return run_type_a<12>(src, dst, p, mv, threads);
    case 8: return run_type_a<8>(src, dst, p, mv, threads);
    case 7: return run_type_a<7>(src, dst, p, mv, threads);
    case 6: return run_type_a<6>(src, dst, p, mv, threads);
    case 5: return run_type_a<5>(src, dst, p, mv, threads);
    case 4: return run_type_a<4>(src, dst, p, mv, threads);
    case 3: return run_type_a<3>(src, dst, p, mv, threads);
    case 2: return run_type_a<2>(src, dst, p, mv, threads);
    default: return run_type_a<1>(src, dst, p, mv, threads);
    }
}

// Copies n bytes; rows of up to 256 bytes in 16-byte pieces (the last one overlapping), which beats calling memcpy().
inline void copy_bytes(uint8_t* d, const uint8_t* s, size_t n) {
    if (n > 256 || n < 16) {
        std::memcpy(d, s, n);
        return;
    }
    for (size_t i = 0; i + 16 < n; i += 16) std::memcpy(d + i, s + i, 16);
    std::memcpy(d + n - 16, s + n - 16, 16);
}

// Type B: the last dimension stays last; row(dst, src, count) moves `count` contiguous elements.
template <class Row>
inline void run_type_b(const uint8_t* src, uint8_t* dst, const Plan& p, size_t ss, size_t ds, int threads,
                       const Row& row) {
    const int n = p.ndim;
    uint64_t sw[kMaxDims] = {}, dw[kMaxDims] = {};
    strides(p, sw, dw);
    Outer o;
    for (int k = 0; k < n - 1; k++) o.add(p.dims[k], sw[k], dw[k]);
    const uint64_t rows = o.count(), len = p.dims[n - 1];
    // Units of work: (row, piece of row). Rows are split only when there are too few of them for the threads.
    uint64_t pieces = 1;
    if (threads > 1 && rows < 4 * static_cast<uint64_t>(threads)) {
        pieces = (4 * static_cast<uint64_t>(threads) + rows - 1) / rows;
        pieces = std::max<uint64_t>(1, std::min<uint64_t>(pieces, len * ss / 4096));
    }
    parallel_for(rows * pieces, threads, [&](uint64_t u0, uint64_t u1) {
        const uint64_t first = u0 / pieces, last = (u1 - 1) / pieces + 1;
        for_outer(o, first, last, [&](uint64_t s, uint64_t d, uint64_t i) {
            const uint64_t q0 = i == first ? u0 % pieces : 0;
            const uint64_t q1 = i == last - 1 ? (u1 - 1) % pieces + 1 : pieces;
            const uint64_t e0 = split(len, pieces, q0), e1 = split(len, pieces, q1);
            row(dst + (d + e0) * ds, src + (s + e0) * ss, e1 - e0);
        });
    });
}

// No permutation left: a copy in pieces.
template <class Row>
inline void run_copy(const uint8_t* src, uint8_t* dst, uint64_t count, size_t ss, size_t ds, int threads,
                     const Row& row) {
    const uint64_t pieces = std::max<uint64_t>(1, std::min<uint64_t>(threads, count * ss / 65536));
    parallel_for(pieces, threads, [&](uint64_t u0, uint64_t u1) {
        const uint64_t e0 = split(count, pieces, u0), e1 = split(count, pieces, u1);
        row(dst + e0 * ds, src + e0 * ss, e1 - e0);
    });
}

}  // namespace detail

/* Permutes the ndim-dimensional row-major tensor src of the given shape into dst: dst has shape
 * (shape[perm[0]], ..., shape[perm[ndim-1]]). Elements are elem_size bytes. num_threads <= 0 picks the number of
 * threads automatically. Returns HPTP_OK or a negative HPTP_ERR_ code; nothing is written on error. */
inline int permute(const void* src, void* dst, size_t elem_size, int ndim, const int64_t* shape, const int64_t* perm,
                   int num_threads = 0) {
    using namespace detail;
    Plan p;
    const int rc = make_plan(ndim, shape, perm, elem_size, elem_size, p);
    if (rc != HPTP_OK || p.numel == 0) return rc;
    if (src == nullptr || dst == nullptr) return HPTP_ERR_NULL;
    const uint64_t bytes = p.numel * elem_size;
    if (overlaps(src, bytes, dst, bytes)) return HPTP_ERR_OVERLAP;
    const int threads = resolve_threads(num_threads, bytes);
    const auto* s = static_cast<const uint8_t*>(src);
    auto* d = static_cast<uint8_t*>(dst);
    const size_t es = elem_size * p.group;
    const auto row = [es](uint8_t* dp, const uint8_t* sp, uint64_t count) { copy_bytes(dp, sp, count * es); };
    if (p.ndim <= 1) {
        const auto copy = [elem_size](uint8_t* dp, const uint8_t* sp, uint64_t count) {
            std::memcpy(dp, sp, count * elem_size);
        };
        run_copy(s, d, p.numel, elem_size, elem_size, threads, copy);
    } else if (p.perm[p.ndim - 1] == p.ndim - 1) {
        run_type_b(s, d, p, es, es, threads, row);
    } else {
        switch (es) {
        case 1: run_type_a<batch_rows(1)>(s, d, p, CopyN<1>(), threads); break;
        case 2: run_type_a<batch_rows(2)>(s, d, p, CopyN<2>(), threads); break;
        case 3: run_type_a<batch_rows(3)>(s, d, p, CopyN<3>(), threads); break;
        case 4: run_type_a<batch_rows(4)>(s, d, p, CopyN<4>(), threads); break;
        case 5: run_type_a<batch_rows(5)>(s, d, p, CopyN<5>(), threads); break;
        case 6: run_type_a<batch_rows(6)>(s, d, p, CopyN<6>(), threads); break;
        case 7: run_type_a<batch_rows(7)>(s, d, p, CopyN<7>(), threads); break;
        case 8: run_type_a<batch_rows(8)>(s, d, p, CopyN<8>(), threads); break;
        case 9: run_type_a<batch_rows(9)>(s, d, p, CopyN<9>(), threads); break;
        case 10: run_type_a<batch_rows(10)>(s, d, p, CopyN<10>(), threads); break;
        case 11: run_type_a<batch_rows(11)>(s, d, p, CopyN<11>(), threads); break;
        case 12: run_type_a<batch_rows(12)>(s, d, p, CopyN<12>(), threads); break;
        case 13: run_type_a<batch_rows(13)>(s, d, p, CopyN<13>(), threads); break;
        case 14: run_type_a<batch_rows(14)>(s, d, p, CopyN<14>(), threads); break;
        case 15: run_type_a<batch_rows(15)>(s, d, p, CopyN<15>(), threads); break;
        case 16: run_type_a<batch_rows(16)>(s, d, p, CopyN<16>(), threads); break;
        default: run_type_a_any(s, d, p, CopyAny{es}, threads); break;
        }
    }
    return HPTP_OK;
}

inline int permute(const void* src, void* dst, size_t elem_size, const std::vector<int64_t>& shape,
                   const std::vector<int64_t>& perm, int num_threads = 0) {
    if (shape.size() != perm.size()) return HPTP_ERR_PERM;
    if (shape.size() > static_cast<size_t>(detail::kMaxDims)) return HPTP_ERR_NDIM;
    return permute(src, dst, elem_size, static_cast<int>(shape.size()), shape.data(), perm.data(), num_threads);
}

/* permute() that also converts each value from TS to TD (see convert()), in one pass over the data. */
template <class TS, class TD>
inline int permute_convert(const TS* src, TD* dst, int ndim, const int64_t* shape, const int64_t* perm,
                           int num_threads = 0) {
    using namespace detail;
    static_assert(std::is_arithmetic<TS>::value && std::is_arithmetic<TD>::value, "arithmetic types only");
    static_assert(!std::is_same<TS, bool>::value && !std::is_same<TD, bool>::value, "bool is not supported");
    if constexpr (std::is_same<TS, TD>::value) {
        return permute(src, dst, sizeof(TS), ndim, shape, perm, num_threads);
    } else {
        Plan p;
        const int rc = make_plan(ndim, shape, perm, std::max(sizeof(TS), sizeof(TD)), sizeof(TD), p);
        if (rc != HPTP_OK || p.numel == 0) return rc;
        if (src == nullptr || dst == nullptr) return HPTP_ERR_NULL;
        if (overlaps(src, p.numel * sizeof(TS), dst, p.numel * sizeof(TD))) return HPTP_ERR_OVERLAP;
        const int threads = resolve_threads(num_threads, p.numel * std::max(sizeof(TS), sizeof(TD)));
        const auto* s = reinterpret_cast<const uint8_t*>(src);
        auto* d = reinterpret_cast<uint8_t*>(dst);
        const uint64_t g = p.group;
        const auto row = [g](uint8_t* dp, const uint8_t* sp, uint64_t count) {
            const TS* a = reinterpret_cast<const TS*>(sp);
            TD* b = reinterpret_cast<TD*>(dp);
            for (uint64_t i = 0, n = count * g; i < n; i++) b[i] = convert<TS, TD>(a[i]);
        };
        if (p.ndim <= 1)
            run_copy(s, d, p.numel, sizeof(TS), sizeof(TD), threads, row);
        else if (p.perm[p.ndim - 1] == p.ndim - 1)
            run_type_b(s, d, p, g * sizeof(TS), g * sizeof(TD), threads, row);
        else if (g == 1)
            run_type_a_any(s, d, p, ConvertOne<TS, TD>(), threads);
        else
            run_type_a_any(s, d, p, ConvertGroup<TS, TD>{g}, threads);
        return HPTP_OK;
    }
}

/* The 2024 interface: shape and permutation as uint64_t arrays, the output shape written to dst_dims. */
inline int permute(const void* src, void* dst, uint64_t dtypeSize, uint64_t* src_dims, uint64_t src_ndim,
                   uint64_t* permute_idx, uint64_t* dst_dims, int nThreads = 1) {
    if (src_ndim > static_cast<uint64_t>(detail::kMaxDims)) return HPTP_ERR_NDIM;
    if (src_ndim > 0 && (src_dims == nullptr || permute_idx == nullptr || dst_dims == nullptr)) return HPTP_ERR_NULL;
    int64_t shape[detail::kMaxDims], perm[detail::kMaxDims];
    const int n = static_cast<int>(src_ndim);
    for (int i = 0; i < n; i++) {
        if (src_dims[i] > static_cast<uint64_t>(INT64_MAX)) return HPTP_ERR_SHAPE;
        shape[i] = static_cast<int64_t>(src_dims[i]);
        perm[i] = permute_idx[i] < src_ndim ? static_cast<int64_t>(permute_idx[i]) : -1;
    }
    const int rc = permute(src, dst, static_cast<size_t>(dtypeSize), n, shape, perm, nThreads);
    if (rc == HPTP_OK)
        for (int i = 0; i < n; i++) dst_dims[i] = src_dims[permute_idx[i]];
    return rc;
}

template <typename T_S, typename T_D>
inline int permute_dtypeConv(const T_S* src, T_D* dst, uint64_t* src_dims, uint64_t src_ndim, uint64_t* permute_idx,
                             uint64_t* dst_dims, int nThreads = 1) {
    if (src_ndim > static_cast<uint64_t>(detail::kMaxDims)) return HPTP_ERR_NDIM;
    if (src_ndim > 0 && (src_dims == nullptr || permute_idx == nullptr || dst_dims == nullptr)) return HPTP_ERR_NULL;
    int64_t shape[detail::kMaxDims], perm[detail::kMaxDims];
    const int n = static_cast<int>(src_ndim);
    for (int i = 0; i < n; i++) {
        if (src_dims[i] > static_cast<uint64_t>(INT64_MAX)) return HPTP_ERR_SHAPE;
        shape[i] = static_cast<int64_t>(src_dims[i]);
        perm[i] = permute_idx[i] < src_ndim ? static_cast<int64_t>(permute_idx[i]) : -1;
    }
    const int rc = permute_convert(src, dst, n, shape, perm, nThreads);
    if (rc == HPTP_OK)
        for (int i = 0; i < n; i++) dst_dims[i] = src_dims[permute_idx[i]];
    return rc;
}

}  // namespace hptp

#endif  // HPTP_HPP
