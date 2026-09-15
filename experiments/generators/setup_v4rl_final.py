"""V4RL в финальной конфигурации: GAT 1 слой 256/256, прогрев 10 эпох случайных негативов, аналог 10/25
(центральная половина интервала ground truth), Qwen3-VL, графы Mask2Former, фильтры мусора и динамики,
исправленное разбиение (части не делят кадры ground truth). Сид 42; слагаемое / без текста / перемешанный текст.
"""
from pathlib import Path

H = Path.home(); C = H / "TGSeqLoc/configs"; DATA = H / "tgseqloc_data"
M2F = "/mnt/external_usb_hdd/6YL/tgseqloc/data/scene_graphs_m2f/{sequence}"
ARMS = {"add": (DATA / "rv4_m2f_qwen", True, False), "notext": (DATA / "rv4_m2f_notext", False, False),
        "shufadd": (DATA / "rv4_m2f_shuf_qwen", True, True)}
base = (C / "v256_set_s42.yaml").read_text().splitlines()
for arm, (root, text, shuffled) in ARMS.items():
    out = []
    for l in base:
        if l.startswith(("  text_fusion:", "  text_dropout:", "  text_centering:")):
            continue
        if l.startswith("  scene_graph_root_template:"):
            l = f"  scene_graph_root_template: {M2F}"
        elif l.startswith("  prepared_root:"):
            l = f"  prepared_root: {root}"
        elif l.startswith("  use_text_nodes:"):
            l = f"  use_text_nodes: {'true' if text else 'false'}"
        elif l.startswith("  epochs:"):
            l = "  epochs: 30"
        elif l.startswith("  patience:"):
            l = "  patience: 5"
        elif l.startswith("  experiment_name:"):
            l = f"  experiment_name: v4final-{arm}-s42"
        elif l == "preprocess:":
            out += ["segmentation:", "  backend: yolo_seg", "  weights: yolo11x_seg", "  params:", "    confidence: 0.25",
                    "text_dynamics:", "  backend: mask_ioa", "  params:", "    threshold: 0.5"]
        out.append(l)
        if l.startswith("  ocr_root_template:"):
            out.append("  ocr_file_name: qwen3_vl_4b.json")
        if l.startswith("  chunk_size:"):
            out.append("  split_guard: ground_truth_overlap")
        if l == "preprocess:":
            out += ["  encoder_batch_size: 256", "  text_min_length: 3", "  text_drop_numeric: true"]
            if shuffled:
                out.append("  shuffle_text_seed: 20260911")
        if l == "  proj_dim: 256":
            out.append("  n_layers: 1")
            if text:
                out.append("  text_fusion: additive")
        if l.startswith("  learning_rate:"):
            out += ["  miner_warmup_epochs: 10", "  positive_interval_fraction: 0.5"]
    out += ["retrieval:", "  top_k: 100"]
    body = "\n".join(out) + "\n"
    for need in ("scene_graphs_m2f", "qwen3_vl_4b.json", "split_guard: ground_truth_overlap", "n_layers: 1", "epochs: 30",
                 "patience: 5", "miner_warmup_epochs: 10", "positive_interval_fraction: 0.5", "text_min_length: 3", "mask_ioa", "top_k: 100"):
        assert need in body, (arm, need)
    assert ("text_fusion: additive" in body) == text and ("shuffle_text_seed" in body) == shuffled
    (C / f"v4final_{arm}_s42.yaml").write_text(body)
print("конфигов: 3")
