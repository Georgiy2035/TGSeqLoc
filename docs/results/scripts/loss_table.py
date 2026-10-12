import json
from pathlib import Path
D = Path.home() / "tgseqloc_data/loss_runs"
for cfg in ("nce_notext_f0", "ncebn_notext_f0", "bn40_notext_f0", "nce_notext_f3", "ncebn_notext_f3", "bn40_notext_f3"):
    p = D / f"{cfg}.json"
    try:
        t = p.read_text(errors="replace"); o, _ = json.JSONDecoder().raw_decode(t[t.find("{"):])
        fm, h = o["final_metrics"], o["history"]
    except Exception:
        print(cfg, "нет результата"); continue
    print(f"{cfg}: лучшая эпоха {fm.get('best_epoch')}, тест R@1 {fm['R@1']:.2f} R@5 {fm['R@5']:.2f} R@10 {fm['R@10']:.2f}")
    print("   loss:   ", " ".join(f"{e['loss']:.3f}" for e in h))
    print("   вал R@5:", " ".join(f"{e['R@5']:.1f}" for e in h))
