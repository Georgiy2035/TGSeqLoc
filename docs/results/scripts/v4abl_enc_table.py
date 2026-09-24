import json, random
from pathlib import Path
import numpy as np
H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
RUNS = {"char n-grams 64 (ref.)": ("v4abl-add_qwen-s42", D / "v4abl_runs/v4abl_add_qwen_s42.json"),
        "char n-grams 384": ("v4abl-enc_cn384-s42", D / "v4abl_enc_runs/v4abl_enc_cn384_s42.json"),
        "E5 small 384": ("v4abl-enc_e5small-s42", D / "v4abl_enc_runs/v4abl_enc_e5small_s42.json"),
        "E5 base 768": ("v4abl-enc_e5base-s42", D / "v4abl_enc_runs/v4abl_enc_e5base_s42.json"),
        "BGE-M3 1024": ("v4abl-enc_bgem3-s42", D / "v4abl_enc_runs/v4abl_enc_bgem3_s42.json"),
        "LaBSE 768": ("v4abl-enc_labse-s42", D / "v4abl_enc_runs/v4abl_enc_labse_s42.json"),
        "no text": ("v4abl-notext-s42", D / "v4abl_runs/v4abl_notext_s42.json")}
KS = (1, 5, 10, 25, 100)
split = json.load(open(D / "abl_v4_qwen/v4rl/mappings/temporal_split.json"))
pos = {int(k): set(v) for k, v in split["positives"].items()}; test = split["test_query_indices"]
hits, rows = {}, {}
for name, (out, log) in RUNS.items():
    r = json.load(open(OUT / out / "test_retrievals.json")); assert r["query_indices"] == test
    hits[name] = {q: [int(any(c in pos[q] for c in near[:k])) for k in KS] for q, near in zip(test, r["nearest_database_indices"])}
    t = open(log).read(); h = json.loads(t[t.index("{"):])["history"]; best = max(h, key=lambda e: e["R@5"])
    rows[name] = dict(rec=[100 * np.mean([hits[name][q][j] for q in test]) for j in range(len(KS))], ep=len(h), best=best["epoch"], val=best["R@5"])
print("%-24s %8s %7s  %s" % ("encoder", "ep/best", "val R@5", " ".join("%-7s" % f"R@{k}" for k in KS)))
for n, r in rows.items():
    print("%-24s %4d/%-3d %7.1f  %s" % (n, r["ep"], r["best"], r["val"], " ".join("%-7.2f" % v for v in r["rec"])))
blocks = [test[i:i+50] for i in range(0, len(test), 50)]; rnd = random.Random(0); ref = "char n-grams 64 (ref.)"
print("\nразницы с эталоном (21 блок по 50 запросов):")
for name in rows:
    if name == ref: continue
    parts = []
    for j, k in enumerate(KS[:3]):
        diff = [np.mean([hits[name][q][j] - hits[ref][q][j] for q in b]) * 100 for b in blocks]; w = [len(b) for b in blocks]
        obs = np.average(diff, weights=w)
        boot = sorted(np.average([diff[i] for i in s], weights=[w[i] for i in s]) for s in ([rnd.randrange(len(blocks)) for _ in blocks] for _ in range(10000)))
        ext = sum(abs(np.average([d * rnd.choice((1, -1)) for d in diff], weights=w)) >= abs(obs) - 1e-12 for _ in range(20000))
        parts.append("R@%d %+.2f [%+.2f; %+.2f] p=%.3f" % (k, obs, boot[250], boot[9750], ext / 20000))
    print("  %-24s %s" % (name, ";  ".join(parts)))
