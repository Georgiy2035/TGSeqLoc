"""Скан lr, RobotCar без текста: кривая валидации и loss по эпохам, лучшая эпоха, тест."""
import json
from pathlib import Path
D = Path.home() / "tgseqloc_data/lr_runs"
def load(p):
    t = open(p, errors="replace").read()
    try:
        o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):]); return o
    except Exception:
        return None
for f in (0, 3):
    print(f"\n=== фолд {f} ===")
    for lr in ("3.0e-5", "1.0e-4", "3.0e-4", "1.0e-3"):
        p = D / f"f{f}_lr{lr}.json"
        o = load(p) if p.exists() else None
        if not o or "history" not in o:
            print(f"  lr {lr}: не готов"); continue
        h = o["history"]; fm = o["final_metrics"]
        print(f"  lr {lr}: лучшая эпоха {fm.get('best_epoch')}, тест R@1 {fm['R@1']:.2f} R@5 {fm['R@5']:.2f} R@10 {fm['R@10']:.2f}")
        print("    loss:    " + " ".join(f"{e['loss']:.3f}" for e in h))
        print("    вал R@5: " + " ".join(f"{e['R@5']:5.1f}" for e in h))
