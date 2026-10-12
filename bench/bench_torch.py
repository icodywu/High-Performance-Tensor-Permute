"""hptpermute against PyTorch on the CPU, in the same process, on the same tensors.

Both sides write into a preallocated output that has already been touched, so the time is the permutation itself:
    torch:  out.copy_(x.permute(p))
    hptp:   hptpermute.permute(x, p, out=out2, num_threads=t)
For reference it also times x.permute(p).contiguous(), which allocates a new tensor each time (page faults included).
Throughput counts bytes read plus bytes written; each figure is the best of at least REPS runs and 0.2 s of runs, so
that short calls are measured once the threads have settled on their cores. Every result is checked against torch
first. Each method is called once untimed and then timed after a 50 ms pause, so that the other library's idle
threads, which spin for a while after a call, have gone to sleep and do not compete for the cores.

usage: python bench/bench_torch.py [--threads 1,8] [--reps 5] [--quick]
"""
import argparse
import platform
import time

import torch

import hptpermute as hp

CASES = [
    # name, shape, dtype, permutations
    ("2-D transpose", (16384, 16384), torch.float32, [(1, 0)]),
    ("2-D transpose", (32768, 32768), torch.uint8, [(1, 0)]),
    ("batched transpose", (64, 1024, 2048), torch.float32, [(0, 2, 1)]),
    ("NCHW -> NHWC", (64, 256, 56, 56), torch.float32, [(0, 2, 3, 1)]),
    ("NHWC -> NCHW", (64, 56, 56, 256), torch.float32, [(0, 3, 1, 2)]),
    ("images CHW -> HWC", (64, 3, 224, 224), torch.uint8, [(0, 2, 3, 1)]),
    ("images HWC -> CHW", (64, 224, 224, 3), torch.uint8, [(0, 3, 1, 2)]),
    ("(B,S,H,D) -> (B,H,S,D)", (32, 2048, 32, 128), torch.bfloat16, [(0, 2, 1, 3)]),
    ("8-D", (12, 32, 8, 16, 24, 16, 20, 3), torch.uint16,
     [(3, 4, 0, 1, 2, 5, 6, 7), (6, 7, 3, 4, 5, 0, 1, 2), (4, 5, 6, 7, 1, 2, 3, 0), (6, 5, 4, 3, 2, 1, 0, 7)]),
    ("8-D", (12, 32, 8, 16, 24, 16, 20, 3), torch.float32,
     [(3, 4, 0, 1, 2, 5, 6, 7), (6, 7, 3, 4, 5, 0, 1, 2), (4, 5, 6, 7, 1, 2, 3, 0), (6, 5, 4, 3, 2, 1, 0, 7)]),
]
QUICK = [("2-D transpose", (4096, 4096), torch.float32, [(1, 0)]),
         ("NCHW -> NHWC", (16, 64, 56, 56), torch.float32, [(0, 2, 3, 1)]),
         ("8-D", (6, 8, 8, 8, 12, 8, 10, 3), torch.uint16, [(6, 7, 3, 4, 5, 0, 1, 2), (6, 5, 4, 3, 2, 1, 0, 7)])]


def best(f, reps, min_seconds=0.2):
    f()
    time.sleep(0.05)
    t, n, start = float("inf"), 0, time.perf_counter()
    while n < reps or time.perf_counter() - start < min_seconds:
        t0 = time.perf_counter()
        f()
        t = min(t, time.perf_counter() - t0)
        n += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,8", help="comma-separated thread counts (torch.set_num_threads and num_threads)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="small tensors, for a smoke test")
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    print("# torch %s, hptpermute %s, %s, %s" % (torch.__version__, hp.library_version(), platform.processor() or platform.machine(),
                                                 platform.platform()))
    print("# GB/s = (bytes read + bytes written) / best time of at least %d runs and 0.2 s; x = hptp / torch copy_"
          % a.reps)
    print("%-24s %-26s %-9s %3s %10s %10s %10s %7s" % ("case", "perm", "dtype", "thr", "contiguous", "copy_", "hptp", "x"))
    g = torch.Generator().manual_seed(0)
    for name, shape, dtype, perms in (QUICK if a.quick else CASES):
        if dtype in (torch.uint8, torch.uint16):
            x = torch.randint(0, 250, shape, dtype=torch.int32, generator=g).to(dtype)
        else:
            x = torch.randn(shape, generator=g).to(dtype)
        gb = 2 * x.numel() * x.element_size() / 1e9
        for p in perms:
            ref = x.permute(p).contiguous()
            out = torch.empty_like(ref).fill_(0)
            out2 = torch.empty_like(ref).fill_(0)
            hp.permute(x, p, out=out2)
            if not torch.equal(out2.view(torch.uint8), ref.view(torch.uint8)):
                raise SystemExit("hptpermute gave a wrong result for %s %s" % (shape, p))
            del ref
            for t in threads:
                torch.set_num_threads(t)
                tc = best(lambda: x.permute(p).contiguous(), a.reps)
                tt = best(lambda: out.copy_(x.permute(p)), a.reps)
                th = best(lambda: hp.permute(x, p, out=out2, num_threads=t), a.reps)
                print("%-24s %-26s %-9s %3d %10.2f %10.2f %10.2f %7.2f" % (
                    name, ",".join(map(str, p)), str(dtype).replace("torch.", ""), t, gb / tc, gb / tt, gb / th, tt / th),
                    flush=True)
            del out, out2


if __name__ == "__main__":
    main()
