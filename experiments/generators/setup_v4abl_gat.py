"""A6/A7 — глубина GAT и размер дескриптора на V4RL, эталон, сид 42.

Меняются только model.n_layers и model.proj_dim; данные те же (abl_v4_qwen),
подготовка не нужна — параметры влияют только на модель.
"""
from pathlib import Path

C = Path.home() / "TGSeqLoc/configs"
SOURCE = C / "v4abl_add_qwen_s42.yaml"
ARMS = {"l2": ("n_layers", 2), "d64": ("proj_dim", 64), "d128": ("proj_dim", 128), "d512": ("proj_dim", 512)}

n = 0
for name, (key, value) in ARMS.items():
    out = []
    for line in SOURCE.read_text().splitlines():
        if line.startswith(f"  {key}:"):
            line = f"  {key}: {value}"
        elif line.startswith("  experiment_name:"):
            line = f"  experiment_name: v4abl-gat_{name}-s42"
        out.append(line)
    body = "\n".join(out) + "\n"
    for need in (f"{key}: {value}", "prepared_root: /home/sitnikov_sv/tgseqloc_data/abl_v4_qwen",
                 "split_guard: ground_truth_blocks", "text_fusion: additive", "ocr_file_name: qwen3_vl_4b.json"):
        assert need in body, (name, need)
    (C / f"v4abl_gat_{name}_s42.yaml").write_text(body)
    n += 1
print("конфигов:", n)
