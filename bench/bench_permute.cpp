// Throughput of hptp::permute() for one shape and permutation.
//
// usage: bench_permute SHAPE PERM ELEM_SIZE [THREADS [REPS [GAP_MS]]]
//   SHAPE and PERM are comma-separated, e.g.  bench_permute 16384,16384 1,0 4 1
//   THREADS 0 (the default) picks the number of threads automatically. GAP_MS > 0 sleeps that long before each run,
//   as for calls that are not back to back (the thread pool's workers are then asleep).
// Prints the best of REPS runs (default 5) into a preallocated, already-touched output, as GB/s of data read plus
// written: 2 * tensor bytes / seconds.
#include "hptp.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static std::vector<int64_t> parse(const char* s) {
    std::vector<int64_t> v;
    std::stringstream in(s);
    std::string item;
    while (std::getline(in, item, ',')) v.push_back(std::stoll(item));
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s SHAPE PERM ELEM_SIZE [THREADS [REPS [GAP_MS]]]\n", argv[0]);
        return 2;
    }
    const std::vector<int64_t> shape = parse(argv[1]), perm = parse(argv[2]);
    const size_t elem = std::strtoul(argv[3], nullptr, 10);
    const int threads = argc > 4 ? std::atoi(argv[4]) : 0;
    const int reps = argc > 5 ? std::atoi(argv[5]) : 5;
    const int gap_ms = argc > 6 ? std::atoi(argv[6]) : 0;
    size_t bytes = elem;
    for (auto d : shape) bytes *= static_cast<size_t>(d);
    std::vector<uint8_t> src(bytes), dst(bytes);
    for (size_t i = 0; i < bytes; i++) src[i] = static_cast<uint8_t>(i * 2654435761u >> 13);
    std::memset(dst.data(), 0, bytes);
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
        if (gap_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(gap_ms));
        const auto t0 = std::chrono::steady_clock::now();
        const int rc = hptp::permute(src.data(), dst.data(), elem, shape, perm, threads);
        const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (rc != HPTP_OK) {
            std::fprintf(stderr, "hptp::permute: %d\n", rc);
            return 1;
        }
        if (t < best) best = t;
    }
    std::printf("shape %s perm %s elem %zu threads %d: %.4f s, %.2f GB/s\n", argv[1], argv[2], elem,
                threads > 0 ? threads : hptp::detail::auto_threads(bytes), best, 2.0 * bytes / best / 1e9);
    return 0;
}
