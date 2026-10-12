// Tests of hptp::permute(), hptp::permute_convert() and the 2024 interface against a direct reference.
//
// usage: test_permute [random_cases [seed]]
// CMake also builds it with tiny HPTP_* blocking sizes (test_permute_small_blocks), so that small tensors go through
// every blocking path.
// Small test tensors would otherwise run on one thread: let every call use the threads it asks for.
#ifndef HPTP_MIN_BYTES_PER_THREAD
#define HPTP_MIN_BYTES_PER_THREAD 1
#endif
#include "hptp.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int failures = 0, checks = 0;

#define CHECK(cond, ...)                                       \
    do {                                                       \
        checks++;                                              \
        if (!(cond)) {                                         \
            failures++;                                        \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);   \
            std::printf(__VA_ARGS__);                          \
            std::printf("\n");                                 \
        }                                                      \
    } while (0)

using Dims = std::vector<int64_t>;

static std::string str(const Dims& v) {
    std::string s = "(";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + ")";
}

static uint64_t numel_of(const Dims& shape) {
    uint64_t n = 1;
    for (auto d : shape) n *= static_cast<uint64_t>(d);
    return n;
}

// dst[i_0..i_{n-1}] = src[j] with j_{perm[k]} = i_k, element by element.
static void reference(const uint8_t* src, uint8_t* dst, size_t es, const Dims& shape, const Dims& perm) {
    const int n = static_cast<int>(shape.size());
    const uint64_t numel = numel_of(shape);
    std::vector<uint64_t> sw(n);
    for (int k = n - 1; k >= 0; k--) sw[k] = k == n - 1 ? 1 : sw[k + 1] * static_cast<uint64_t>(shape[k + 1]);
    std::vector<int64_t> idx(n, 0);
    for (uint64_t j = 0; j < numel; j++) {
        uint64_t off = 0;
        for (int i = 0; i < n; i++) off += static_cast<uint64_t>(idx[i]) * sw[perm[i]];
        std::memcpy(dst + j * es, src + off * es, es);
        for (int i = n - 1; i >= 0; i--) {
            if (++idx[i] < shape[perm[i]]) break;
            idx[i] = 0;
        }
    }
}

// Buffers of exactly the tensor size, so that AddressSanitizer sees any access outside them.
struct Buf {
    std::unique_ptr<uint8_t[]> p;
    size_t n;
    explicit Buf(size_t bytes) : p(bytes ? new uint8_t[bytes] : nullptr), n(bytes) {}
    uint8_t* data() { return p.get(); }
};

static void run_case(const Dims& shape, const Dims& perm, size_t es, int threads, std::mt19937_64& rng) {
    const uint64_t numel = numel_of(shape);
    const size_t bytes = numel * es;
    Buf src(bytes), dst(bytes), ref(bytes);
    for (size_t i = 0; i < bytes; i++) src.data()[i] = static_cast<uint8_t>(rng());
    if (bytes) std::memset(dst.data(), 0xEE, bytes);
    reference(src.data(), ref.data(), es, shape, perm);
    const int rc = hptp::permute(src.data(), dst.data(), es, static_cast<int>(shape.size()), shape.data(),
                                 perm.data(), threads);
    CHECK(rc == HPTP_OK, "shape %s perm %s elem %zu threads %d: rc %d", str(shape).c_str(), str(perm).c_str(), es,
          threads, rc);
    CHECK(bytes == 0 || std::memcmp(dst.data(), ref.data(), bytes) == 0, "shape %s perm %s elem %zu threads %d: wrong output",
          str(shape).c_str(), str(perm).c_str(), es, threads);
}

static void named_cases(std::mt19937_64& rng) {
    struct C {
        Dims shape, perm;
    };
    const std::vector<C> cases = {
        {{}, {}},                                       // a scalar
        {{0}, {0}},                                     // empty
        {{3, 0, 4}, {2, 0, 1}},                         // empty
        {{10}, {0}},                                    // 1-D
        {{2, 3, 4}, {0, 1, 2}},                         // identity
        {{1, 5, 1, 7}, {2, 1, 0, 3}},                   // only size-1 dimensions move
        {{4, 1, 6}, {0, 2, 1}},                         // a transpose next to a size-1 dimension
        {{5, 1, 4}, {2, 1, 0}},                         //
        {{1, 1, 1}, {2, 0, 1}},                         // a single element
        {{3, 5}, {1, 0}},                               // small transpose
        {{6, 4, 5}, {1, 0, 2}},                         // last dimension kept and folded
        {{2, 3, 4, 5}, {3, 2, 1, 0}},                   // reversal
        {{8, 9, 12, 6, 4, 6, 9, 10}, {2, 3, 4, 5, 0, 1, 6, 7}},  // merges into 3 dimensions
        {{20, 8, 16, 12, 6}, {2, 1, 3, 0, 4}},          // fold of 6 values
        {{20, 8, 16, 12, 17}, {2, 1, 3, 0, 4}},         // last dimension too large to fold: memcpy rows
        {{300, 1100}, {1, 0}},                          // transpose larger than one block
        {{1031, 517}, {1, 0}},                          //
        {{513, 300}, {1, 0}},                           // wide elements: tiles of a few columns
        {{5, 7001}, {1, 0}},                            // few long rows
        {{7001, 5}, {1, 0}},                            // many short rows
        {{3, 257, 129}, {0, 2, 1}},                     // batched transpose
        {{2, 67, 13, 11}, {0, 2, 3, 1}},                // NCHW -> NHWC
        {{2, 13, 11, 67}, {0, 3, 1, 2}},                // NHWC -> NCHW
        {{2, 33, 5, 16}, {0, 2, 1, 3}},                 // (B, S, H, D) -> (B, H, S, D)
        {{2, 3, 50001}, {1, 0, 2}},                     // few rows to split between threads
        {{3, 4, 5, 6, 7}, {4, 3, 2, 1, 0}},             //
        {{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, {11, 0, 10, 1, 9, 2, 8, 3, 7, 4, 6, 5}},
    };
    const size_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 33};
    const int threads[] = {1, 3, 8, 0};
    for (const auto& c : cases)
        for (size_t es : sizes)
            for (int t : threads) {
                if (numel_of(c.shape) * es > (8u << 20)) continue;
                run_case(c.shape, c.perm, es, t, rng);
            }
}

static void random_cases(int count, std::mt19937_64& rng) {
    const size_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 20, 24, 32, 33};
    for (int c = 0; c < count; c++) {
        Dims shape, perm;
        uint64_t numel;
        do {
            const int ndim = static_cast<int>(rng() % 9);
            shape.assign(ndim, 1);
            for (auto& d : shape) {
                const unsigned r = static_cast<unsigned>(rng() % 100);
                d = r < 30 ? 1 : r < 32 ? 0 : r < 90 ? 2 + static_cast<int64_t>(rng() % 6) : 8 + static_cast<int64_t>(rng() % 60);
            }
            numel = numel_of(shape);
        } while (numel > 40000);
        perm.resize(shape.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::shuffle(perm.begin(), perm.end(), rng);
        const size_t es = sizes[rng() % (sizeof(sizes) / sizeof(sizes[0]))];
        const int threads = static_cast<int>(rng() % 10);  // 0 = automatic
        run_case(shape, perm, es, threads, rng);
    }
}

// Several threads of the caller permute at once: one gets the thread pool, the others start their own threads.
static void concurrent_callers() {
    std::vector<std::thread> callers;
    std::vector<int> bad(6, 0);
    for (int k = 0; k < 6; k++)
        callers.emplace_back([k, &bad] {
            std::mt19937_64 rng(100 + k);
            const Dims shapes[] = {{37, 45, 3}, {64, 3, 50}, {5, 6, 7, 8}};
            const Dims perms[] = {{1, 0, 2}, {0, 2, 1}, {3, 1, 0, 2}};
            for (int round = 0; round < 30; round++) {
                const Dims& shape = shapes[round % 3];
                const Dims& perm = perms[round % 3];
                const size_t es = 1 + static_cast<size_t>(rng() % 9), bytes = numel_of(shape) * es;
                std::vector<uint8_t> src(bytes), dst(bytes), ref(bytes);
                for (auto& v : src) v = static_cast<uint8_t>(rng());
                reference(src.data(), ref.data(), es, shape, perm);
                const int rc = hptp::permute(src.data(), dst.data(), es, static_cast<int>(shape.size()), shape.data(),
                                             perm.data(), 2 + k % 4);
                if (rc != HPTP_OK || dst != ref) bad[k]++;
            }
        });
    for (auto& t : callers) t.join();
    for (int k = 0; k < 6; k++) CHECK(bad[k] == 0, "concurrent caller %d: %d wrong results", k, bad[k]);
}

static void inverse_roundtrip(std::mt19937_64& rng) {
    // permute(permute(x, p), inverse(p)) == x, through the 2024 interface, which also reports the output shape.
    std::vector<uint64_t> shape = {4, 5, 6, 8, 9, 10, 12, 7}, perm = {4, 5, 6, 0, 1, 2, 3, 7};
    for (int round = 0; round < 20; round++) {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        std::vector<uint16_t> a(n), b(n), c(n);
        for (uint64_t i = 0; i < n; i++) a[i] = static_cast<uint16_t>(i * 2654435761u);
        std::vector<uint64_t> inv(8), outShape(8), backShape(8);
        for (int i = 0; i < 8; i++) inv[perm[i]] = i;
        int rc = hptp::permute(a.data(), b.data(), 2, shape.data(), 8, perm.data(), outShape.data(), 1 + round % 8);
        CHECK(rc == 0, "legacy permute rc %d", rc);
        for (int i = 0; i < 8; i++) CHECK(outShape[i] == shape[perm[i]], "legacy dst_dims[%d]", i);
        rc = hptp::permute(b.data(), c.data(), 2, outShape.data(), 8, inv.data(), backShape.data(), 1 + round % 5);
        CHECK(rc == 0 && a == c && backShape == shape, "legacy round trip %d", round);
        std::shuffle(perm.begin(), perm.end(), rng);
        for (auto& d : shape) d = 1 + rng() % 7;
    }
}

static void errors() {
    uint8_t a[64] = {}, b[64];
    std::memset(b, 0x5A, sizeof(b));
    const Dims s2 = {4, 8}, p2 = {1, 0};
    const int64_t bad_perm[] = {0, 0}, out_of_range[] = {0, 2}, neg_shape[] = {4, -1};
    const int64_t huge[] = {int64_t(1) << 40, int64_t(1) << 40};
    std::vector<int64_t> big(HPTP_MAX_DIMS + 1, 1), bigp(HPTP_MAX_DIMS + 1);
    std::iota(bigp.begin(), bigp.end(), 0);
    CHECK(hptp::permute(a, b, 1, -1, s2.data(), p2.data()) == HPTP_ERR_NDIM, "ndim -1");
    CHECK(hptp::permute(a, b, 1, HPTP_MAX_DIMS + 1, big.data(), bigp.data()) == HPTP_ERR_NDIM, "ndim too large");
    CHECK(hptp::permute(a, b, 1, 2, s2.data(), bad_perm) == HPTP_ERR_PERM, "repeated axis");
    CHECK(hptp::permute(a, b, 1, 2, s2.data(), out_of_range) == HPTP_ERR_PERM, "axis out of range");
    CHECK(hptp::permute(a, b, 1, 2, neg_shape, p2.data()) == HPTP_ERR_SHAPE, "negative size");
    CHECK(hptp::permute(a, b, 1, 2, huge, p2.data()) == HPTP_ERR_SHAPE, "size overflow");
    CHECK(hptp::permute(a, b, 0, 2, s2.data(), p2.data()) == HPTP_ERR_ELEM, "elem_size 0");
    CHECK(hptp::permute(nullptr, b, 1, 2, s2.data(), p2.data()) == HPTP_ERR_NULL, "null src");
    CHECK(hptp::permute(a, nullptr, 1, 2, s2.data(), p2.data()) == HPTP_ERR_NULL, "null dst");
    CHECK(hptp::permute(a, b, 1, 2, nullptr, p2.data()) == HPTP_ERR_NULL, "null shape");
    CHECK(hptp::permute(a, a + 8, 1, 2, s2.data(), p2.data()) == HPTP_ERR_OVERLAP, "overlap");
    CHECK(hptp::permute(a, a, 1, 2, s2.data(), p2.data()) == HPTP_ERR_OVERLAP, "same buffer");
    bool untouched = true;
    for (uint8_t v : b) untouched = untouched && v == 0x5A;
    CHECK(untouched, "dst written on error");
    const Dims e = {0, 5}, pe = {1, 0};
    CHECK(hptp::permute(nullptr, nullptr, 4, 2, e.data(), pe.data()) == HPTP_OK, "empty tensor with null pointers");
    CHECK(hptp::permute(a, b, 1, {4, 8}, {1, 0, 2}) == HPTP_ERR_PERM, "vector sizes differ");
    std::vector<uint64_t> ls = {4, 8}, lp = {1, 1}, ld(2);
    CHECK(hptp::permute(a, b, 1, ls.data(), 2, lp.data(), ld.data()) == HPTP_ERR_PERM, "legacy bad perm");
}

// ---- permute_convert ----

template <class TS, class TD>
static void convert_case(const Dims& shape, const Dims& perm, int threads, std::mt19937_64& rng) {
    const uint64_t n = numel_of(shape);
    std::vector<TS> src(n);
    std::vector<TD> conv(n), ref(n), dst(n, TD(7));
    for (auto& v : src) {
        if constexpr (std::is_floating_point<TS>::value)
            v = static_cast<TS>((static_cast<double>(rng() % 2000001) - 1000000.0) * 1e-1);
        else
            v = static_cast<TS>(rng());
    }
    for (uint64_t i = 0; i < n; i++) conv[i] = hptp::convert<TS, TD>(src[i]);
    reference(reinterpret_cast<const uint8_t*>(conv.data()), reinterpret_cast<uint8_t*>(ref.data()), sizeof(TD), shape,
              perm);
    const int rc = hptp::permute_convert(src.data(), dst.data(), static_cast<int>(shape.size()), shape.data(), perm.data(),
                                         threads);
    CHECK(rc == HPTP_OK && std::memcmp(dst.data(), ref.data(), n * sizeof(TD)) == 0,
          "permute_convert %zu->%zu bytes shape %s perm %s threads %d rc %d", sizeof(TS), sizeof(TD), str(shape).c_str(),
          str(perm).c_str(), threads, rc);
}

template <class TS, class TD>
static void convert_cases(std::mt19937_64& rng) {
    const std::vector<std::pair<Dims, Dims>> cases = {
        {{}, {}},
        {{7}, {0}},
        {{3, 4, 5}, {0, 1, 2}},
        {{37, 53}, {1, 0}},
        {{6, 5, 3}, {1, 0, 2}},          // a folded last dimension of 3 values
        {{4, 6, 2, 9}, {2, 0, 3, 1}},
        {{5, 7, 40}, {1, 0, 2}},         // rows copied whole
        {{3, 129, 67}, {0, 2, 1}},
        {{2, 9, 1, 11, 3}, {3, 1, 0, 2, 4}},
    };
    for (const auto& c : cases)
        for (int t : {1, 4})
            convert_case<TS, TD>(c.first, c.second, t, rng);
}

static void convert_values() {
    using hptp::convert;
    // volatile: keeps compilers from folding the out-of-range casts that convert() guards against
    volatile double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    volatile double big = 1e300;
    CHECK((convert<int32_t, int8_t>(300) == 127 && convert<int32_t, int8_t>(-300) == -128), "int32 -> int8");
    CHECK((convert<int32_t, int8_t>(-5) == -5 && convert<int32_t, int16_t>(70000) == 32767), "int32 -> int8/int16");
    CHECK((convert<int64_t, int32_t>(int64_t(1) << 40) == INT32_MAX), "int64 -> int32");
    CHECK((convert<int32_t, uint32_t>(-1) == 0 && convert<uint32_t, int32_t>(4000000000u) == INT32_MAX), "sign change");
    CHECK((convert<uint8_t, int8_t>(200) == 127 && convert<int8_t, uint64_t>(-3) == 0), "uint8 -> int8, int8 -> uint64");
    CHECK((convert<int8_t, int64_t>(-5) == -5 && convert<uint16_t, uint8_t>(65535) == 255), "widening, narrowing unsigned");
    CHECK((convert<double, float>(big) == static_cast<float>(inf) && convert<double, float>(-big) == -static_cast<float>(inf)),
          "double -> float overflow");
    volatile float tenth = convert<double, float>(0.1);  // stored, so x87 excess precision is rounded away
    CHECK((std::isnan(convert<double, float>(nan)) && tenth == 0.1f), "double -> float");
    CHECK((convert<float, int32_t>(3e9f) == INT32_MAX && convert<float, int32_t>(-3e9f) == INT32_MIN), "float -> int32 clamp");
    CHECK((convert<float, int32_t>(2.9f) == 2 && convert<float, int32_t>(-2.9f) == -2), "float -> int32 truncates");
    CHECK((convert<double, int32_t>(nan) == 0 && convert<float, uint8_t>(-1.5f) == 0 && convert<float, uint8_t>(300.f) == 255),
          "NaN and unsigned clamps");
    CHECK((convert<float, int64_t>(9.3e18f) == INT64_MAX && convert<double, uint64_t>(1.9e19) == UINT64_MAX), "64-bit clamps");
    CHECK((convert<uint64_t, float>(UINT64_MAX) == 18446744073709551616.0f && convert<int32_t, double>(-7) == -7.0), "to float");
}

static void legacy_convert() {
    std::vector<int32_t> a(2 * 3 * 4);
    for (size_t i = 0; i < a.size(); i++) a[i] = static_cast<int32_t>(i * 50) - 400;
    std::vector<int8_t> b(a.size());
    std::vector<uint64_t> shape = {2, 3, 4}, perm = {2, 0, 1}, out(3);
    const int rc = hptp::permute_dtypeConv(a.data(), b.data(), shape.data(), 3, perm.data(), out.data(), 2);
    CHECK(rc == 0 && out == std::vector<uint64_t>({4, 2, 3}), "legacy permute_dtypeConv rc %d", rc);
    // b[k][i][j] = clamp(a[i][j][k])
    bool ok = true;
    for (int k = 0; k < 4; k++)
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++) ok = ok && b[(k * 2 + i) * 3 + j] == hptp::convert<int32_t, int8_t>(a[(i * 3 + j) * 4 + k]);
    CHECK(ok, "legacy permute_dtypeConv values");
}

int main(int argc, char** argv) {
    const int count = argc > 1 ? std::atoi(argv[1]) : 1500;
    const unsigned long long seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
    std::mt19937_64 rng(seed);
    named_cases(rng);
    random_cases(count, rng);
    concurrent_callers();
    inverse_roundtrip(rng);
    errors();
    convert_values();
    convert_cases<int32_t, int8_t>(rng);
    convert_cases<int32_t, int16_t>(rng);
    convert_cases<int64_t, int32_t>(rng);
    convert_cases<double, float>(rng);
    convert_cases<float, double>(rng);
    convert_cases<uint8_t, float>(rng);
    convert_cases<float, int32_t>(rng);
    convert_cases<int16_t, int64_t>(rng);
    convert_cases<uint16_t, int8_t>(rng);
    convert_cases<float, float>(rng);
    legacy_convert();
    std::printf("%d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
