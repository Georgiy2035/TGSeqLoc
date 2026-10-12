"""Абляции V4RL на эталонной конфигурации с новым разбиением (docs/ablations_v4rl.md).

Эталон — v4final_add_s42 (GAT 1 слой 256/256, прогрев 10 эпох, центральная половина интервала, Qwen3-VL,
Mask2Former, фильтры мусора и динамики). Меняется разбиение: валидация — два блока с центрами на 30 % и 60 %
аннотированных запросов до теста (split_guard: ground_truth_blocks), тест — хвост целиком.

Сейчас: A1 — модель OCR. Сид 42.
  add_qwen    — Qwen3-VL 4B (эталон)
  add_paddle  — PaddleOCR v5
  add_easyocr — EasyOCR
  notext      — без текста
"""
from pathlib import Path

H = Path.home()
C = H / "TGSeqLoc/configs"
DATA = H / "tgseqloc_data"
CENTRES = "[0.3, 0.6]"
ARMS = {
    "add_qwen": ("v4final_add_s42.yaml", "qwen3_vl_4b.json", DATA / "abl_v4_qwen"),
    "add_paddle": ("v4final_add_s42.yaml", "paddleocr_v5.json", DATA / "abl_v4_paddle"),
    "add_easyocr": ("v4final_add_s42.yaml", "easyocr.json", DATA / "abl_v4_easyocr"),
    "notext": ("v4final_notext_s42.yaml", "qwen3_vl_4b.json", DATA / "abl_v4_notext"),
}
SEEDS = (42,)

n = 0
for arm, (source, ocr, root) in ARMS.items():
    for seed in SEEDS:
        out = []
        for line in (C / source).read_text().splitlines():
            if line.startswith("  ocr_file_name:"):
                line = f"  ocr_file_name: {ocr}"
            elif line.startswith("  prepared_root:"):
                line = f"  prepared_root: {root}"
            elif line.startswith("  split_guard:"):
                out += ["  split_guard: ground_truth_blocks", f"  validation_centres: {CENTRES}"]
                continue
            elif line.startswith("  seed:"):
                line = f"  seed: {seed}"
            elif line.startswith("  experiment_name:"):
                line = f"  experiment_name: v4abl-{arm}-s{seed}"
            out.append(line)
        body = "\n".join(out) + "\n"
        for need in ("split_guard: ground_truth_blocks", f"validation_centres: {CENTRES}", f"ocr_file_name: {ocr}",
                     "scene_graphs_m2f", "n_layers: 1", "miner_warmup_epochs: 10", "positive_interval_fraction: 0.5",
                     "mask_ioa", "top_k: 100", str(root)):
            assert need in body, (arm, need)
        assert ("use_text_nodes: false" in body) == (arm == "notext")
        assert ("text_fusion: additive" in body) == (arm != "notext")
        (C / f"v4abl_{arm}_s{seed}.yaml").write_text(body)
        n += 1
print("конфигов:", n)
