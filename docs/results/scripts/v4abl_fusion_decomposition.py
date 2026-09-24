import json, random, sys
from pathlib import Path
import numpy as np
D = Path.home() / "tgseqloc_data"; G = Path("/mnt/external_usb_hdd/6YL/tgseqloc/outputs"); OUT = Path.home() / "TGSeqLoc/outputs"
R = {"m2f+add (ref)": OUT / "v4abl-add_qwen-s42", "m2f+set": G / "v4a3-set-s42",
     "m2fpost+set": G / "v4best-m2fpost-set-qwen-s42", "gdino+set": G / "v4best-gdino-set-qwen-s42"}
KS = (1, 5, 10)
split = json.load(open(D / "abl_v4_qwen/v4rl/mappings/temporal_split.json"))
pos = {int(k): set(v) for k, v in split["positives"].items()}; test = split["test_query_indices"]
hits = {}
for n, d in R.items():
    r = json.load(open(d / "test_retrievals.json")); assert r["query_indices"] == test
    hits[n] = {q: [int(any(c in pos[q] for c in near[:k])) for k in KS] for q, near in zip(test, r["nearest_database_indices"])}
blocks = [test[i:i+50] for i in range(0, len(test), 50)]; rnd = random.Random(0)
def cmp(a, b):
    parts = []
    for j, k in enumerate(KS):
        diff = [np.mean([hits[a][q][j] - hits[b][q][j] for q in bl]) * 100 for bl in blocks]; w = [len(x) for x in blocks]
        obs = np.average(diff, weights=w)
        ext = sum(abs(np.average([d * rnd.choice((1, -1)) for d in diff], weights=w)) >= abs(obs) - 1e-12 for _ in range(20000))
        parts.append("R@%d %+.2f p=%.3f" % (k, obs, ext / 20000))
    print("  %-28s %s" % (a + " − " + b, ";  ".join(parts)))
for n in R: print("%-14s %s" % (n, " ".join("%-7.2f" % (100*np.mean([hits[n][q][j] for q in test])) for j in range(3))))
print("\nразложение по факторам:")
cmp("m2f+set", "m2f+add (ref)"); cmp("gdino+set", "m2f+set"); cmp("m2fpost+set", "m2f+set"); cmp("gdino+set", "m2f+add (ref)")
