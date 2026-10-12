"""Размер GAT на RobotCar против базовой конфигурации (hidden 128, дескриптор 64)."""
import glob, itertools, json, math, os
from pathlib import Path

H = Path.home()
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}


def load(pattern):
    out = {}
    for f in glob.glob(str(pattern)):
        txt = open(f, encoding="utf-8", errors="replace").read()
        try:
            obj, _ = json.JSONDecoder().raw_decode(txt[txt.find("{"):])
        except Exception:
            continue
        m = obj.get("final_metrics") or obj.get("training", {}).get("final_metrics")
        if m:
            out[int(os.path.basename(f)[:-5].rsplit("_", 1)[1])] = (m["R@1"], m["R@5"], m["R@10"])
    return out


base = {"notext": load(H / "tgseqloc_data/rcg_runs_filter_off/notext_*.json"),
        "real_paddle": load(H / "tgseqloc_data/rcgf_runs/real_paddle_*.json"),
        "real_qwen": load(H / "tgseqloc_data/rcgf_runs/real_qwen_*.json")}
runs = {("h128p64", a): r for a, r in base.items()}
for v in ("h256p64", "h256p256"):
    for a in base:
        runs[(v, a)] = load(H / f"tgseqloc_data/gsz_runs/{v}_{a}_*.json")


def ci(v):
    n = len(v); mu = sum(v) / n
    s = math.sqrt(sum((x - mu) ** 2 for x in v) / (n - 1)) if n > 1 else 0
    return "%5.2f ± %5.2f" % (mu, T.get(n, 2.262) * s / math.sqrt(n) if n > 1 else 0)


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]; n = len(d); o = sum(d) / n
    hits = sum(1 for s in itertools.product((1, -1), repeat=n) if abs(sum(si * x for si, x in zip(s, d)) / n) >= abs(o) - 1e-12)
    return o, hits / 2 ** n, sum(1 for x in d if x > 0), n


names = {"notext": "без текста", "real_paddle": "PaddleOCR, фильтр", "real_qwen": "Qwen, фильтр"}
print("=== RobotCar, размер GAT (hidden / дескриптор), 10 сидов ===")
print("%-10s %-18s %3s %-15s %-15s %-15s" % ("GAT", "ветка", "n", "R@1", "R@5", "R@10"))
for v in ("h128p64", "h256p64", "h256p256"):
    for a in base:
        r = runs[(v, a)]
        if not r:
            print("%-10s %-18s  —" % (v, names[a])); continue
        s = sorted(r)
        print("%-10s %-18s %3d %s  %s  %s" % (v, names[a], len(s), *(ci([r[x][k] for x in s]) for k in range(3))))
print("\nпарные сравнения, R@5 (R@1, R@10):")
cmp = [((v, a), ("h128p64", a)) for v in ("h256p64", "h256p256") for a in base]
cmp += [((v, a), (v, "notext")) for v in ("h256p64", "h256p256") for a in ("real_paddle", "real_qwen")]
for x, y in cmp:
    s = sorted(set(runs[x]) & set(runs[y]))
    if len(s) < 2:
        continue
    res = [paired([runs[x][i][k] for i in s], [runs[y][i][k] for i in s]) for k in range(3)]
    print("  %-8s %-18s − %-8s %-18s R@5 Δ=%+6.2f p=%.3f %d/%d | R@1 Δ=%+6.2f p=%.3f | R@10 Δ=%+6.2f p=%.3f"
          % (x[0], names[x[1]], y[0], names[y[1]], res[1][0], res[1][1], res[1][2], res[1][3], res[0][0], res[0][1], res[2][0], res[2][1]))
