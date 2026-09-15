import json
from pathlib import Path
D = Path.home() / "tgseqloc_data/g1_runs"
for f in (0, 3):
    print(f"=== фолд {f} ===")
    for v, label in (("hard", "hard mining (контроль)"), ("rand", "А: случайные негативы"), ("warm", "Б: 10 случайных → hard")):
        try:
            t = (D / f"g1_{v}_f{f}.json").read_text(errors="replace"); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
            fm, h = o["final_metrics"], o["history"]
        except Exception:
            print(f"  {label}: нет результата"); continue
        best = max(h, key=lambda e: e["R@5"])
        print(f"  {label:24s} лучшая эпоха {fm.get('best_epoch')} (вал R@5 {best['R@5']:.1f}), тест R@1 {fm['R@1']:.2f} R@5 {fm['R@5']:.2f} R@10 {fm['R@10']:.2f}")
        print("     loss:   ", " ".join(f"{e['loss']:.3f}" for e in h))
        print("     вал R@5:", " ".join(f"{e['R@5']:.1f}" for e in h))
