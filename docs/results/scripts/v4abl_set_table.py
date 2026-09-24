import json, random
from pathlib import Path
import numpy as np
D = Path.home() / "tgseqloc_data"; OUT = Path.home() / "TGSeqLoc/outputs"
RUNS = {"Ref.: GATv2 1/4/256/256": "v4abl-set_ref-s42", "Layers 2": "v4abl-set_l2-s42",
        "Descriptor 64": "v4abl-set_d64-s42", "Descriptor 128": "v4abl-set_d128-s42",
        "Descriptor 512": "v4abl-set_d512-s42",
        "Ref. seed 43": "v4abl-set_ref-s43", "Ref. seed 44": "v4abl-set_ref-s44",
        "Additive seed 42": "v4abl-add_qwen-s42", "Additive seed 43": "v4abl-add_ref-s43",
        "Additive seed 44": "v4abl-add_ref-s44"}
KS = (1, 5, 10)
split = json.load(open(D / "abl_v4_qwen/v4rl/mappings/temporal_split.json"))
pos = {int(k): set(v) for k, v in split["positives"].items()}; test = split["test_query_indices"]
hits, rows = {}, {}
for name, o in RUNS.items():
    p = OUT / o / "test_retrievals.json"
    if not p.exists():
        print("НЕТ:", name); continue
    r = json.load(open(p)); assert r["query_indices"] == test, name
    hits[name] = {q: [int(any(c in pos[q] for c in near[:k])) for k in KS] for q, near in zip(test, r["nearest_database_indices"])}
    rows[name] = [100 * np.mean([hits[name][q][j] for q in test]) for j in range(len(KS))]
print("%-26s %s" % ("вариант", " ".join("%-7s" % f"R@{k}" for k in KS)))
for n, r in rows.items():
    print("%-26s %s" % (n, " ".join("%-7.2f" % v for v in r)))
blocks = [test[i:i+50] for i in range(0, len(test), 50)]; rnd = random.Random(0); ref = "Ref.: GATv2 1/4/256/256"
print("\nразницы с эталоном:")
for name in rows:
    if name == ref or name not in hits: continue
    parts = []
    for j, k in enumerate(KS):
        diff = [np.mean([hits[name][q][j] - hits[ref][q][j] for q in b]) * 100 for b in blocks]; w = [len(b) for b in blocks]
        obs = np.average(diff, weights=w)
        ext = sum(abs(np.average([d * rnd.choice((1, -1)) for d in diff], weights=w)) >= abs(obs) - 1e-12 for _ in range(20000))
        parts.append("R@%d %+.2f p=%.3f" % (k, obs, ext / 20000))
    print("  %-26s %s" % (name, ";  ".join(parts)))
for arm, names in (("набор строк", ["Ref.: GATv2 1/4/256/256", "Ref. seed 43", "Ref. seed 44"]),
                   ("слагаемое", ["Additive seed 42", "Additive seed 43", "Additive seed 44"])):
    got = [rows[n] for n in names if n in rows]
    if len(got) == 3:
        a = np.array(got)
        print("%-12s по 3 сидам: %s" % (arm, " ".join("R@%d %.2f±%.2f" % (KS[j], a[:, j].mean(), a[:, j].std(ddof=1)) for j in range(3))))
