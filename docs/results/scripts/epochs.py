"""На какой эпохе выбирается лучший результат: упирается ли обучение в потолок 8 эпох."""
import collections, json
from pathlib import Path

H = Path.home() / "tgseqloc_data"
groups = {"RobotCar без текста 256": "f256_runs/notext_f*_*.json", "RobotCar слагаемое 256": "f256_runs/add_f*_*.json",
          "RobotCar слагаемое 10/25": "f256_runs/addp10_f*_*.json", "RobotCar набор строк 256": "f256_runs/set_f*_*.json",
          "RobotCar перемешанный набор": "f256_runs/shufset_f*_*.json", "RobotCar узел 128/64": "cv_runs/real_qwen_f*_*.json",
          "V4RL без текста 256": "v256_runs/notext_*.json", "V4RL узел 256": "v256_runs/node_*.json",
          "V4RL набор строк 256": "v256_runs/set_*.json"}


def history(p):
    try:
        t = open(p, errors="replace").read(); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
        tr = o.get("training", o)
        return tr.get("history") if (tr.get("final_metrics") or o.get("final_metrics")) else None
    except Exception:
        return None


print("%-30s %3s  %-26s %-13s %-16s %s" % ("ветка", "n", "лучшая эпоха: число прогонов", "лучшая = 8-я", "дошло до 8 эпох", "рост вал. R@5 от 1-й к лучшей"))
for name, pat in groups.items():
    hs = [h for p in sorted(H.glob(pat)) if (h := history(p))]
    if not hs:
        continue
    best = [max(h, key=lambda e: e.get("R@5", -1))["epoch"] for h in hs]
    c = collections.Counter(best)
    print("%-30s %3d  %-26s %-13s %-16s %.2f" % (
        name, len(hs), " ".join(f"{k}:{c[k]}" for k in sorted(c)),
        "%d (%.0f %%)" % (c[8], 100 * c[8] / len(hs)),
        "%d (%.0f %%)" % (sum(h[-1]["epoch"] == 8 for h in hs), 100 * sum(h[-1]["epoch"] == 8 for h in hs) / len(hs)),
        sum(max(e.get("R@5", 0) for e in h) - h[0].get("R@5", 0) for h in hs) / len(hs)))
