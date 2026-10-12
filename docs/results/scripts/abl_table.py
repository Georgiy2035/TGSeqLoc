import json
from pathlib import Path
H = Path.home() / "tgseqloc_data"
def load(p):
    try:
        t = p.read_text(errors="replace"); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):]); return o["final_metrics"], o["history"]
    except Exception:
        return None
for f in (0, 3):
    print(f"=== фолд {f} ===")
    for label, p in (("как сейчас", H / f"lr_runs/f{f}_lr3.0e-5.json"), ("dropout 0", H / f"abl_runs/abl_drop0_f{f}.json"),
                     ("случайные негативы", H / f"abl_runs/abl_rand_f{f}.json"), ("батч 64", H / f"abl_runs/abl_b64_f{f}.json")):
        r = load(p)
        if not r:
            print(f"  {label}: нет результата"); continue
        fm, h = r
        print(f"  {label:20s} лучшая эпоха {fm.get('best_epoch')}, тест R@1 {fm['R@1']:.2f} R@5 {fm['R@5']:.2f} R@10 {fm['R@10']:.2f}")
        print("     loss:   ", " ".join(f"{e['loss']:.3f}" for e in h))
        print("     вал R@5:", " ".join(f"{e['R@5']:.1f}" for e in h))
