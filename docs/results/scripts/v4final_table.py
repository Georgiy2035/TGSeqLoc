"""V4RL, финальная конфигурация, сид 42: R@1/5/10/25/100 на тесте; интервалы — бутстреп по отрезкам теста
(соседние запросы по 50 штук, т. к. соседние кадры зависимы), p — перестановочный знаковый тест по отрезкам."""
import json, random
from pathlib import Path

H = Path.home(); D = H / "tgseqloc_data"; OUT = H / "TGSeqLoc/outputs"
KS = (1, 5, 10, 25, 100)
rnd = random.Random(0)
ARMS = {"слагаемое (Qwen)": ("add", "rv4_m2f_qwen"), "без текста": ("notext", "rv4_m2f_notext"), "перемешанный текст": ("shufadd", "rv4_m2f_shuf_qwen")}
res = {}
for name, (key, root) in ARMS.items():
    split = json.load(open(D / root / "v4rl/mappings/temporal_split.json"))
    pos = {int(k): set(v) for k, v in split["positives"].items()}
    r = json.load(open(OUT / f"v4final-{key}-s42" / "test_retrievals.json"))
    assert len(r["nearest_database_indices"][0]) >= 100
    fm = json.load(open(OUT / f"v4final-{key}-s42" / "final_metrics.json"))
    t = open(D / f"v4final_runs/{key}_s42.json").read(); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
    hits = {q: [int(any(c in pos[q] for c in near[:k])) for k in KS] for q, near in zip(r["query_indices"], r["nearest_database_indices"])}
    res[name] = (hits, fm, o)
queries = sorted(next(iter(res.values()))[0])
chunks = [queries[i:i + 50] for i in range(0, len(queries), 50)]
print(f"=== V4RL, финальная конфигурация, сид 42; тест {len(queries)} запросов, {len(chunks)} отрезков по 50 ===")
print("%-20s %5s %8s " % ("вариант", "эпоха", "вал R@5") + "  ".join("%-20s" % f"R@{k}" for k in KS))
for name, (hits, fm, o) in res.items():
    cells = []
    for j in range(len(KS)):
        val = lambda smp: 100 * sum(hits[q][j] for c in smp for q in c) / sum(len(c) for c in smp)
        bs = sorted(val([rnd.choice(chunks) for _ in chunks]) for _ in range(3000))
        cells.append("%5.2f [%5.2f; %5.2f]" % (val(chunks), bs[75], bs[2924]))
    print("%-20s %5s %8.1f " % (name, fm.get("best_epoch"), max(e["R@5"] for e in o["history"])) + "  ".join("%-20s" % c for c in cells))
names = list(ARMS)
for x, y in ((names[0], names[1]), (names[0], names[2]), (names[2], names[1])):
    parts = []
    for j, k in enumerate(KS):
        d = [100 * sum(res[x][0][q][j] - res[y][0][q][j] for q in c) / len(c) for c in chunks]; w = [len(c) for c in chunks]
        obs = sum(a * b for a, b in zip(d, w)) / sum(w)
        ext = sum(1 for _ in range(10000) if abs(sum(a * b * (1 if rnd.random() < .5 else -1) for a, b in zip(d, w)) / sum(w)) >= abs(obs) - 1e-12) / 10000
        parts.append("R@%d %+.2f (p=%.4f)" % (k, obs, ext))
    print("  %s − %s: %s" % (x, y, "; ".join(parts)))
for name, (hits, fm, o) in res.items():
    print("\n%s: loss %s" % (name, " ".join("%.3f" % e["loss"] for e in o["history"])))
    print("%s  вал R@5 %s" % (" " * len(name), " ".join("%.1f" % e["R@5"] for e in o["history"])))
