"""Сводка: время на кадр по этапам (медиана / p95 / среднее), загрузка GPU и CPU и видеопамять в окне этапа."""
import csv, json, statistics, sys
from pathlib import Path
d = Path(sys.argv[1])
rows = [json.loads(l) for l in open(d / "stages.jsonl")]
mon = [dict(t=float(r["t"]), gpu=float(r["gpu_util"]), mem=float(r["gpu_mem_mb"]), cpu=float(r["cpu_percent"])) for r in csv.DictReader(open(d / "monitor.csv"))]
tg = next((r for r in rows if r["stage"] == "_tgseqloc_window"), None)
idle = [m for m in mon if m["t"] < rows[0]["start"]] if rows and rows[0].get("start") else []
base_mem = min(m["mem"] for m in mon) if mon else 0
print("| Этап | Устройство | Кадров | мс/кадр: медиана | p95 | среднее | GPU загрузка, %% (средн./макс.) | Видеопамять, МБ (макс., с учётом базового %d) | Пик torch, МБ | CPU, %% всех ядер (средн.) |" % base_mem)
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["stage"].startswith("_"):
        continue
    start, end = (r["start"], r["end"]) if r.get("start") else (tg["start"], tg["end"])
    w = [m for m in mon if start <= m["t"] <= end]
    t = sorted(r["times_ms"]); p95 = t[int(0.95 * (len(t) - 1))]
    gpu = "%.0f / %.0f" % (statistics.mean(m["gpu"] for m in w), max(m["gpu"] for m in w)) if w else "—"
    mem = "%.0f" % max(m["mem"] for m in w) if w else "—"
    cpu = "%.0f" % statistics.mean(m["cpu"] for m in w) if w else "—"
    shared = " (окно общее для этапов TGSeqLoc)" if r.get("shared_window") else ""
    peak = "%.0f" % r["torch_peak_mb"] if r.get("torch_peak_mb") else "—"
    print("| %s | %s | %d | %.1f | %.1f | %.1f | %s | %s | %s | %s%s |" % (r["stage"], r["device"], len(t), statistics.median(t), p95, statistics.mean(t), gpu, mem, peak, cpu, shared))
tg_rows = [r for r in rows if r.get("shared_window")]
if tg_rows:
    per = [sum(x) for x in zip(*[r["times_ms"] for r in tg_rows])]
    print("\nTGSeqLoc после OCR и масок (фильтр + кодирование + граф + GAT + поиск): медиана %.1f мс, p95 %.1f мс на кадр" % (statistics.median(per), sorted(per)[int(0.95 * (len(per) - 1))]))
