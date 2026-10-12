import json, random
from pathlib import Path
import numpy as np
D = Path.home() / "tgseqloc_data"; OUT = Path.home() / "TGSeqLoc/outputs"; G = Path("/mnt/external_usb_hdd/6YL/tgseqloc/outputs")
RUNS = {"слагаемое (старый эталон)": (OUT / "v4abl-add_qwen-s42", D / "v4abl_runs/v4abl_add_qwen_s42.json"),
        "набор строк (новый эталон)": (OUT / "v4abl-set_ref-s42", D / "v4abl_set_runs/v4abl_set_ref_s42.json"),
        "  + перемешанный текст": (OUT / "v4abl-set_shuf-s42", D / "v4abl_set_runs/v4abl_set_shuf_s42.json"),
        "  + 2 слоя GAT": (OUT / "v4abl-set_l2-s42", D / "v4abl_set_runs/v4abl_set_l2_s42.json"),
        "  + дескриптор 128": (OUT / "v4abl-set_d128-s42", D / "v4abl_set_runs/v4abl_set_d128_s42.json"),
        "  + дескриптор 512": (OUT / "v4abl-set_d512-s42", D / "v4abl_set_runs/v4abl_set_d512_s42.json"),
        "без текста": (OUT / "v4abl-notext-s42", D / "v4abl_runs/v4abl_notext_s42.json"),
        "чужой набор строк (a3_set)": (G / "v4a3-set-s42", None)}
KS = (1, 5, 10, 25, 100)
split = json.load(open(D / "abl_v4_qwen/v4rl/mappings/temporal_split.json"))
pos = {int(k): set(v) for k, v in split["positives"].items()}; test = split["test_query_indices"]
hits, rows = {}, {}
for name, (d, log) in RUNS.items():
    p = d / "test_retrievals.json"
    if not p.exists():
        print("нет", name); continue
    r = json.load(open(p)); assert r["query_indices"] == test, name
    hits[name] = {q: [int(any(c in pos[q] for c in near[:k])) for k in KS] for q, near in zip(test, r["nearest_database_indices"])}
    val, ep, best = float("nan"), 0, 0
    if log and Path(log).exists():
        t = open(log).read(); h = json.loads(t[t.index("{"):])["history"]; b = max(h, key=lambda e: e["R@5"])
        val, ep, best = b["R@5"], len(h), b["epoch"]
    rows[name] = dict(rec=[100 * np.mean([hits[name][q][j] for q in test]) for j in range(len(KS))], ep=ep, best=best, val=val)
print("%-28s %8s %7s  %s" % ("вариант", "эп/лучш", "вал R@5", " ".join("%-7s" % f"R@{k}" for k in KS)))
for n, r in rows.items():
    print("%-28s %4d/%-3d %7.1f  %s" % (n, r["ep"], r["best"], r["val"], " ".join("%-7.2f" % v for v in r["rec"])))
blocks = [test[i:i+50] for i in range(0, len(test), 50)]; rnd = random.Random(0); ref = "набор строк (новый эталон)"
print("\nразницы с новым эталоном (21 блок по 50 запросов):")
for name in rows:
    if name == ref: continue
    parts = []
    for j, k in enumerate(KS[:3]):
        diff = [np.mean([hits[name][q][j] - hits[ref][q][j] for q in b]) * 100 for b in blocks]; w = [len(b) for b in blocks]
        obs = np.average(diff, weights=w)
        boot = sorted(np.average([diff[i] for i in s], weights=[w[i] for i in s]) for s in ([rnd.randrange(len(blocks)) for _ in blocks] for _ in range(10000)))
        ext = sum(abs(np.average([d * rnd.choice((1, -1)) for d in diff], weights=w)) >= abs(obs) - 1e-12 for _ in range(20000))
        parts.append("R@%d %+.2f [%+.2f; %+.2f] p=%.3f" % (k, obs, boot[250], boot[9750], ext / 20000))
    print("  %-28s %s" % (name, ";  ".join(parts)))
