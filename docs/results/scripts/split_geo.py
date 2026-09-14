"""Разбиение RobotCar: география, текст и связь одного с другим."""
import collections, json, math, statistics as st
from pathlib import Path
import torch

H = Path.home()
ROOT = H / "tgseqloc_data/rcgf_real_paddle/robotcar"   # ветка PaddleOCR с фильтром
INS = {"base": "/mnt/external_usb_hdd/6YL/Datasets/robotcar/2015-08-12-15-04-18/gps/ins.csv",
       "query": "/mnt/external_usb_hdd/6YL/Datasets/robotcar/2015-09-02-10-37-32/gps/ins.csv"}
OUT = Path("/tmp/geo"); OUT.mkdir(exist_ok=True)

from tgseqloc.data.robotcar import load_ins_track

s = json.load(open(ROOT / "mappings/temporal_split.json"))
tracks = {k: load_ins_track(Path(v)) for k, v in INS.items()}


def info(rel, seq):
    stem = Path(rel).stem
    g = torch.load(ROOT / rel, weights_only=False)
    strs = [t.strip() for t in (getattr(g, "text_strings", None) or []) if t.strip()]
    return tracks[seq].position_at(float(stem)), strs


db = [info(rel, "base") for rel in s["database_paths"]]
q = [info(rel, "query") for rel in s["query_paths"]]
sets = {"обучение": s["train_query_indices"], "валидация": s["validation_query_indices"], "тест": s["test_query_indices"]}
excluded = s.get("excluded_test_indices", [])

print("=== размеры ===")
print("  база %d кадров | запросы %d" % (len(db), len(q)))
for name, idx in sets.items():
    print("  %-10s %5d запросов" % (name, len(idx)))
print("  исключены из теста как повторный заезд: %d" % len(excluded))


def geo(points):
    pts = [p for p in points if p]
    if not pts:
        return None
    n = [p[0] for p in pts]; e = [p[1] for p in pts]
    path = sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))
    return dict(n=(min(n), max(n)), e=(min(e), max(e)), path=path, pts=pts)


print("\n=== география (UTM, метры) ===")
gdb = geo([p for p, _ in db])
print("  база: путь %.0f м, север %.0f..%.0f, восток %.0f..%.0f" % (gdb["path"], *gdb["n"], *gdb["e"]))
zones = {}
for name, idx in sets.items():
    g = geo([q[i][0] for i in idx])
    zones[name] = g
    print("  %-10s путь %6.0f м, север %.0f..%.0f, восток %.0f..%.0f" % (name, g["path"], *g["n"], *g["e"]))


def mind(a, b):
    return min(math.dist(x, y) for x in a for y in b)


tr, va, te = (zones[k]["pts"] for k in ("обучение", "валидация", "тест"))
print("  ближайшее расстояние тест↔обучение: %.0f м | тест↔валидация: %.0f м | валидация↔обучение: %.0f м"
      % (mind(te[::3], tr[::3]), mind(te[::3], va), mind(va, tr[::3])))

print("\n=== текст по частям разбиения ===")
def texts(idx):
    got = [q[i][1] for i in idx]
    return got


rows = []
for name, idx in sets.items():
    t = texts(idx)
    with_text = sum(1 for x in t if x)
    nodes = sum(len(x) for x in t)
    rows.append((name, len(idx), with_text, nodes))
    print("  %-10s кадров с текстом %4d из %4d (%4.1f %%), текстовых узлов %4d, узлов на кадр %.2f"
          % (name, with_text, len(idx), 100 * with_text / len(idx), nodes, nodes / len(idx)))
dbt = [x for _, x in db]
print("  %-10s кадров с текстом %4d из %4d (%4.1f %%), узлов %4d" % ("база", sum(1 for x in dbt if x), len(dbt), 100 * sum(1 for x in dbt if x) / len(dbt), sum(len(x) for x in dbt)))

print("\n  частые строки:")
for name, idx in sets.items():
    c = collections.Counter(w.lower() for i in idx for w in q[i][1])
    print("    %-10s %s" % (name, ", ".join(f"{w}({n})" for w, n in c.most_common(8)) or "—"))
c = collections.Counter(w.lower() for x in dbt for w in x)
print("    %-10s %s" % ("база", ", ".join(f"{w}({n})" for w, n in c.most_common(8))))

print("\n=== сколько базы приходится на тестовую зону ===")
pos = s["positives"]
test_db = {p for i in sets["тест"] for p in pos[str(i)]}
print("  кадров базы, отвечающих хотя бы одному тестовому запросу: %d из %d" % (len(test_db), len(db)))
print("  из них с текстом: %d (%.1f %%)" % (sum(1 for i in test_db if db[i][1]), 100 * sum(1 for i in test_db if db[i][1]) / len(test_db)))
print("  позитивов на тестовый запрос: медиана %.0f" % st.median(len(pos[str(i)]) for i in sets["тест"]))

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(17, 8))
    dbp = [p for p, _ in db if p]
    for a in ax:
        a.scatter([p[1] for p in dbp], [p[0] for p in dbp], s=3, c="lightgrey", label="база (карта)")
    colors = {"обучение": "tab:blue", "валидация": "tab:orange", "тест": "tab:green"}
    for name, idx in sets.items():
        pts = [q[i][0] for i in idx if q[i][0]]
        ax[0].scatter([p[1] for p in pts], [p[0] for p in pts], s=6, c=colors[name], label=f"{name} ({len(idx)})")
        with_t = [q[i][0] for i in idx if q[i][0] and q[i][1]]
        ax[1].scatter([p[1] for p in with_t], [p[0] for p in with_t], s=14, c=colors[name], label=f"{name} с текстом ({len(with_t)})")
    exc = [q[i][0] for i in excluded if q[i][0]]
    if exc:
        ax[0].scatter([p[1] for p in exc], [p[0] for p in exc], s=26, c="red", marker="x", label=f"исключены ({len(exc)})")
    for a, t in zip(ax, ("разбиение запросов по географии", "только кадры, где остался текст")):
        a.set_aspect("equal"); a.legend(loc="best", fontsize=8); a.grid(alpha=.3); a.set_title(t)
        a.set_xlabel("восток, м"); a.set_ylabel("север, м")
    fig.tight_layout(); fig.savefig(OUT / "split.png", dpi=90)
    print("\nкарта: /tmp/geo/split.png")
except Exception as e:
    print("карта не построена:", type(e).__name__, e)
