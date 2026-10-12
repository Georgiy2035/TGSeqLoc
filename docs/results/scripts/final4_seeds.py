"""Финальный эксперимент, слагаемое: метрики по сидам (все фолды вместе) и по фолдам × сидам."""
import json, statistics, sys
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10, 25, 100)
key, root = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("add", "rcg4_m2f_add_paddle")
cam_of = lambda p: Path(p).stem.split("-", 1)[0] if "-" in Path(p).stem else "stereo_centre"

per = {}  # (fold, seed) -> list of hit tuples
for f in range(5):
    split = json.load(open(D / root / f"robotcar/mappings/split_fold{f}.json"))
    pos = {int(k): set(v) for k, v in split["positives"].items()}
    for s in (42, 43, 44):
        p = OUT / f"final4-{key}-f{f}-s{s}" / "test_retrievals.json"
        if not p.exists():
            continue
        r = json.load(open(p))
        per[(f, s)] = [tuple(int(any(c in pos.get(q, ()) for c in near[:k])) for k in KS)
                       for q, near in zip(r["query_indices"], r["nearest_database_indices"])]

rate = lambda hits: [100 * sum(h[j] for h in hits) / len(hits) for j in range(len(KS))]
print(f"=== {key}: по сидам (все 5 фолдов, среднее по запросам) ===")
print("сид   " + "  ".join("%7s" % f"R@{k}" for k in KS) + "   запросов")
seed_rows = {}
for s in (42, 43, 44):
    hits = [h for (f, ss), v in per.items() if ss == s for h in v]
    if not hits:
        continue
    seed_rows[s] = rate(hits)
    print("%-5d " % s + "  ".join("%7.2f" % x for x in seed_rows[s]) + "   %d" % len(hits))
if len(seed_rows) > 1:
    cols = list(zip(*seed_rows.values()))
    print("размах" + "  ".join("%7.2f" % (max(c) - min(c)) for c in cols))
    print("sd    " + "  ".join("%7.2f" % statistics.stdev(c) for c in cols))

print(f"\n=== {key}: R@1 / R@5 / R@10 по фолдам × сидам ===")
print("фолд  " + "  ".join("%-22s" % f"сид {s}" for s in (42, 43, 44)) + "  размах R@5 между сидами")
for f in range(5):
    cells, r5 = [], []
    for s in (42, 43, 44):
        if (f, s) in per:
            x = rate(per[(f, s)]); r5.append(x[1])
            cells.append("%5.2f / %5.2f / %5.2f" % (x[0], x[1], x[2]))
        else:
            cells.append("—")
    print("%-5d " % f + "  ".join("%-22s" % c for c in cells) + "  %.2f" % (max(r5) - min(r5) if r5 else float("nan")))
fold_means = [statistics.mean(rate(per[(f, s)])[1] for s in (42, 43, 44) if (f, s) in per) for f in range(5)]
print("\nR@5 средний по сидам для каждого фолда:", ", ".join("%.2f" % x for x in fold_means),
      "→ разброс между фолдами sd %.2f" % statistics.stdev(fold_means))
within = [statistics.stdev([rate(per[(f, s)])[1] for s in (42, 43, 44)]) for f in range(5) if all((f, s) in per for s in (42, 43, 44))]
print("разброс между сидами внутри фолда (sd R@5), в среднем по фолдам: %.2f" % statistics.mean(within))
