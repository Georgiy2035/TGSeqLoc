"""Сид 42: метрики по каждому фолду для четырёх вариантов (тройки с любой / той же камеры × слагаемое / без текста)."""
import json, statistics
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10, 25, 100)
cam_of = lambda p: Path(p).stem.split("-", 1)[0] if "-" in Path(p).stem else "stereo_centre"
ARMS = [("любая / слагаемое", "final4", "add", "rcg4_m2f_add_paddle"), ("любая / без текста", "final4", "notext", "rcg4_m2f_notext"),
        ("та же / слагаемое", "final4s", "add", "rcg4_m2f_add_paddle"), ("та же / без текста", "final4s", "notext", "rcg4_m2f_notext")]
res = {}
for name, pref, key, root in ARMS:
    for f in range(5):
        split = json.load(open(D / root / f"robotcar/mappings/split_fold{f}.json"))
        pos = {int(k): set(v) for k, v in split["positives"].items()}
        dcam = [cam_of(p) for p in split["database_paths"]]; qp = split["query_paths"]
        r = json.load(open(OUT / f"{pref}-{key}-f{f}-s42" / "test_retrievals.json"))
        a, b = [], []
        for q, near in zip(r["query_indices"], r["nearest_database_indices"]):
            pq = pos.get(q, set()); same = {j for j in pq if dcam[j] == cam_of(qp[q])}
            a.append([int(any(c in pq for c in near[:k])) for k in KS])
            if same:
                b.append([int(any(c in same for c in near[:k])) for k in KS])
        ep = json.load(open(OUT / f"{pref}-{key}-f{f}-s42" / "final_metrics.json")).get("best_epoch")
        res[(name, f)] = ([100 * sum(h[j] for h in a) / len(a) for j in range(len(KS))],
                          [100 * sum(h[j] for h in b) / len(b) for j in range(len(KS))], len(a), ep)
for mi, title in ((0, "метрика а (любая камера ≤ 25 м)"), (1, "метрика б (та же камера ≤ 25 м)")):
    print(f"\n=== {title}, сид 42 ===")
    print("%-5s %-20s %6s %5s  " % ("фолд", "вариант", "запр.", "эпоха") + "  ".join("%6s" % f"R@{k}" for k in KS))
    for f in range(5):
        for name, *_ in ARMS:
            m, _, n, ep = res[(name, f)][mi], None, res[(name, f)][2], res[(name, f)][3]
            print("%-5d %-20s %6d %5s  " % (f, name, n, ep) + "  ".join("%6.2f" % x for x in res[(name, f)][mi]))
        print()
    print("простое среднее 5 фолдов:")
    for name, *_ in ARMS:
        cols = list(zip(*[res[(name, f)][mi] for f in range(5)]))
        print("      %-20s %6s %5s  " % (name, "", "") + "  ".join("%6.2f" % statistics.mean(c) for c in cols))
    print("разница «та же − любая» по фолдам, R@5:")
    for arm in ("слагаемое", "без текста"):
        d = [res[(f"та же / {arm}", f)][mi][1] - res[(f"любая / {arm}", f)][mi][1] for f in range(5)]
        print("      %-12s " % arm + "  ".join("%+6.2f" % x for x in d))
    print("разница «слагаемое − без текста» по фолдам, R@5:")
    for tr in ("любая", "та же"):
        d = [res[(f"{tr} / слагаемое", f)][mi][1] - res[(f"{tr} / без текста", f)][mi][1] for f in range(5)]
        print("      %-12s " % (tr + " камера") + "  ".join("%+6.2f" % x for x in d))
