"""The README's performance table from outputs of bench_torch.py.

usage: python bench/make_table.py bench/results/epyc9334_torch2.14.1_*.txt

Each cell is PyTorch's GB/s -> hptpermute's GB/s (speedup). When several files have the same case and thread count,
each library gets its best figure among them.
"""
import collections
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

rows = collections.OrderedDict()
for f in sys.argv[1:]:
    for line in open(f):
        m = re.match(r"(.{24}) (\S+)\s+(\S+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$", line.rstrip())
        if not m or line.startswith("case") or line.startswith("#"):
            continue
        name, perm, dt, thr, cont, cp, hp, x = m.groups()
        cell = rows.setdefault((name.strip(), perm, dt), {})
        old = cell.get(int(thr))
        new = (float(cont), float(cp), float(hp))
        cell[int(thr)] = new if old is None else tuple(max(a, b) for a, b in zip(old, new))

SHAPES = {"2-D transpose": {"float32": "16384×16384", "uint8": "32768×32768"},
          "batched transpose": "64×1024×2048", "NCHW -> NHWC": "64×256×56×56",
          "NHWC -> NCHW": "64×56×56×256", "images CHW -> HWC": "64×3×224×224",
          "images HWC -> CHW": "64×224×224×3", "(B,S,H,D) -> (B,H,S,D)": "32×2048×32×128",
          "8-D": "12×32×8×16×24×16×20×3"}
NAMES = {"2-D transpose": "2-D transpose", "batched transpose": "batched transpose", "NCHW -> NHWC": "NCHW → NHWC",
         "NHWC -> NCHW": "NHWC → NCHW", "images CHW -> HWC": "images, CHW → HWC",
         "images HWC -> CHW": "images, HWC → CHW", "(B,S,H,D) -> (B,H,S,D)": "(B, S, H, D) → (B, H, S, D)"}


def num(v):
    return "%.0f" % v if v >= 100 else "%.1f" % v if v >= 1 else "%.2f" % v


threads = sorted({t for r in rows.values() for t in r})
print("| Permutation | Shape | dtype | " + " | ".join("%d thread%s" % (t, "" if t == 1 else "s") for t in threads) + " |")
print("|---|---|---|" + "---:|" * len(threads))
for (name, perm, dt), r in rows.items():
    shape = SHAPES[name][dt] if isinstance(SHAPES[name], dict) else SHAPES[name]
    label = NAMES.get(name) or "8-D, (%s)" % perm.replace(",", ", ")
    cells = []
    for t in threads:
        if t not in r:
            cells.append("")
            continue
        cont, cp, hp = r[t]
        x = hp / cp
        xs = "%.0f" % x if x >= 10 else "%.1f" % x if x >= 0.95 else "%.2f" % x
        cells.append("%s → %s (%s×)" % (num(cp), num(hp), xs))
    print("| %s | %s | %s | %s |" % (label, shape, dt, " | ".join(cells)))
