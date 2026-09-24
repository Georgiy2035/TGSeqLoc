"""Перепрогон на новой опорной конфигурации: набор строк (text_fusion: set), V4RL, сид 42.

База — v4abl_add_qwen_s42 (M2F, Qwen3-VL, фильтры, GAT 1x4 256/256, блочная валидация),
меняется только форма объединения текста и один параметр на арм. Подготовка не нужна:
text_fusion — параметр модели, корни данных переиспользуются.
"""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
SOURCE = C / "v4abl_add_qwen_s42.yaml"
SHUF = "/mnt/external_usb_hdd/6YL/tgseqloc/prepared/a3_shuf"
# арм: (ключ, значение) или None для опорного; отдельным полем — свой корень данных
ARMS = {"ref": (None, None), "shuf": (None, None), "l2": ("n_layers", 2),
        "d128": ("proj_dim", 128), "d512": ("proj_dim", 512), "d64": ("proj_dim", 64)}
ROOTS = {"shuf": SHUF}

n = 0
for name, (key, value) in ARMS.items():
    out = []
    for line in SOURCE.read_text().splitlines():
        if line.startswith("  text_fusion:"):
            line = "  text_fusion: set"
        elif key and line.startswith(f"  {key}:"):
            line = f"  {key}: {value}"
        elif name in ROOTS and line.startswith("  prepared_root:"):
            line = f"  prepared_root: {ROOTS[name]}"
        elif line.startswith("  experiment_name:"):
            line = f"  experiment_name: v4abl-set_{name}-s42"
        out.append(line)
    body = "\n".join(out) + "\n"
    needs = ["text_fusion: set", "split_guard: ground_truth_blocks", "ocr_file_name: qwen3_vl_4b.json", "seed: 42"]
    if key:
        needs.append(f"{key}: {value}")
    needs.append(f"prepared_root: {ROOTS.get(name, '/home/sitnikov_sv/tgseqloc_data/abl_v4_qwen')}")
    for need in needs:
        assert need in body, (name, need)
    assert body.count("text_fusion:") == 1
    (C / f"v4abl_set_{name}_s42.yaml").write_text(body)
    n += 1
print("конфигов:", n)
