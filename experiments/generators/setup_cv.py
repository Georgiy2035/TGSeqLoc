"""Конфиги блочной кросс-валидации: 3 ветки × 5 фолдов × 5 сидов."""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
FOLDS_FILE = "/home/sitnikov_sv/tgseqloc_data/robotcar_folds.json"
ARMS = {"notext": "rcg_notext", "real_qwen": "rcgf_real_qwen", "shuffled_qwen": "rcgf_shuffled_qwen"}
n = 0
for arm, src_prefix in ARMS.items():
    for fold in range(5):
        for seed in range(42, 47):
            t = (C / f"{src_prefix}_s{seed}.yaml").read_text(encoding="utf-8")
            anchor = next(l for l in t.splitlines() if l.strip().startswith("frame_list_path:"))
            t = t.replace(anchor + "\n",
                          anchor + f"\n  split_folds_path: {FOLDS_FILE}\n  split_fold: {fold}\n", 1)
            old = f"experiment_name: {src_prefix.replace('_', '-', 1)}-s{seed}"
            assert t.count(old) == 1, (src_prefix, seed)
            t = t.replace(old, f"experiment_name: cv-{arm}-f{fold}-s{seed}")
            (C / f"cv_{arm}_f{fold}_s{seed}.yaml").write_text(t, encoding="utf-8")
            n += 1
print("конфигов:", n)
