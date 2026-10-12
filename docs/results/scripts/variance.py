"""Разброс по фолдам против разброса по сидам: RobotCar, кросс-валидация 128/64, R@5."""
import itertools, json, math
from pathlib import Path

H = Path.home() / "tgseqloc_data"


def m(p):
    try:
        t = open(p, errors="replace").read(); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
        x = o.get("final_metrics") or o.get("training", {}).get("final_metrics"); return x["R@5"] if x else None
    except Exception:
        return None


arms = {"без текста": "cv_runs/notext_f{f}_{s}.json", "узел": "cv_runs/real_qwen_f{f}_{s}.json",
        "слагаемое": "add_runs/rc_p0_f{f}_{s}.json", "перемешанный (узел)": "cv_runs/shuffled_qwen_f{f}_{s}.json"}
R = {a: {(f, s): m(H / p.format(f=f, s=s)) for f in range(5) for s in range(42, 47)} for a, p in arms.items()}
sd = lambda v: math.sqrt(sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1))
print("%-20s %-8s %-22s %-22s %-10s %-10s" % ("ветка", "среднее", "± если 25 независимы", "± по 5 фолдам", "sd фолдов", "sd сидов"))
for a, r in R.items():
    v = list(r.values()); fm = [sum(r[(f, s)] for s in range(42, 47)) / 5 for f in range(5)]
    within = math.sqrt(sum(sd([r[(f, s)] for s in range(42, 47)]) ** 2 for f in range(5)) / 5)
    print("%-20s %6.2f   ± %5.2f                ± %5.2f                %5.2f      %5.2f" % (a, sum(v) / 25, 2.064 * sd(v) / 5, 2.776 * sd(fm) / math.sqrt(5), sd(fm), within))
print("\nразницы между ветками по фолдам (минимально возможное p при 5 фолдах — 0.0625):")
for x, y in (("слагаемое", "узел"), ("слагаемое", "без текста"), ("узел", "перемешанный (узел)"), ("без текста", "перемешанный (узел)")):
    fd = [sum(R[x][(f, s)] - R[y][(f, s)] for s in range(42, 47)) / 5 for f in range(5)]; mu = sum(fd) / 5
    p = sum(1 for g in itertools.product((1, -1), repeat=5) if abs(sum(a * b for a, b in zip(g, fd)) / 5) >= abs(mu) - 1e-12) / 32
    print("  %-34s Δ=%+5.2f ± %4.2f  фолдов в плюс %d/5  p=%.3f  | %s" % (x + " − " + y, mu, 2.776 * sd(fd) / math.sqrt(5), sum(v > 0 for v in fd), p, "  ".join("%+.2f" % v for v in fd)))
