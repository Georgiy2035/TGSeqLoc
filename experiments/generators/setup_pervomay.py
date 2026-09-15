"""Первомайская без обучения: веса финального RobotCar (4 камеры, 10/25, прогрев, сид 42, фолды 0–4).

Данные: запросы — прогулка 27-08-26, база — 28-08-26, каждый 10-й кадр; графы Mask2Former; PaddleOCR v5;
фильтры мусора и динамики (YOLO11x-seg, IoA ≥ 0.5) как на RobotCar; словарь классов и отношений — из
манифеста RobotCar, классы, которых там нет, читаются как unknown. Правильный кадр — ≤ 25 м (3D).

Корни: текст PaddleOCR, без текста, перемешанный текст (тот же seed, что на RobotCar).
Оценки (веса final4 и final4s, фолды 0–4):
  add     — веса со слагаемым, текст;
  notext  — веса без текста, графы без текста;
  textoff — веса со слагаемым, графы без текста;
  shuf    — веса со слагаемым, перемешанный текст.

  python setup_pervomay.py masks     — маски YOLO в stages/segmentation трёх корней
  python setup_pervomay.py configs   — конфиги подготовки и оценок
"""
import json
import sys
from pathlib import Path

import numpy as np

H = Path.home()
C = H / "TGSeqLoc/configs"
DATA = H / "tgseqloc_data"
SEG = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/pervomay_seg")
WALKS = "/mnt/external_usb_hdd/6YL/kartashov_ga_tmp/pervomay-street-dataset"
ROLE = {"28-08-26": "base", "27-08-26": "query"}
ROOTS = {"add": DATA / "pm_m2f_paddle", "notext": DATA / "pm_m2f_notext", "shuf": DATA / "pm_m2f_shuf_paddle"}
VOCABULARY = DATA / "rcg4_m2f_add_paddle/robotcar/manifest.json"
KEEP = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/tgseqloc_checkpoints")
SHUFFLE_SEED = 20260911
# (оценка, корень, веса)
EVALS = {"add": ("add", "add"), "notext": ("notext", "notext"), "textoff": ("notext", "add"), "shuf": ("shuf", "add")}

DATASET = f"""dataset:
  adapter: pervomay
  root: {WALKS}
  sequences: [base, query]
  traversals:
    base: 28-08-26
    query: 27-08-26
  ocr_root_template: /mnt/external_usb_hdd/6YL/Datasets/pervomay_ocr/{{traversal}}
  ocr_file_name: paddleocr_v5.json
  scene_graph_root_template: /mnt/external_usb_hdd/6YL/Datasets/scene_graphs_pervomay_m2f/{{traversal}}
  image_path_template: {WALKS}/{{traversal}}
  poses_path_template: {WALKS}/{{traversal}}/poses.csv
  gt_path: {WALKS}/28-08-26/poses.csv
  gt_radius_m: 25.0
  prepared_root: {{root}}
  reference_sequence: base
  query_sequence: query
"""


def masks():
    from pycocotools import mask as mask_utils

    front = DATA / "rcg4_m2f_add_paddle/robotcar/stages/segmentation/base"
    identity = json.loads(next(front.glob("*.json")).read_text())["model_identity"]
    for root in ROOTS.values():
        written = 0
        for walk, role in ROLE.items():
            target_dir = root / "pervomay/stages/segmentation" / role
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted((SEG / walk).glob("*/yolo11x_seg.json")):
                target = target_dir / f"{src.parent.name}.json"
                if target.exists():
                    continue
                d = json.loads(src.read_text())
                h, w = d["image_height"], d["image_width"]
                rle = None
                if d["instances"]:
                    union = np.zeros((h, w), np.uint8)
                    for inst in d["instances"]:
                        union |= mask_utils.decode({"size": inst["rle"]["size"], "counts": inst["rle"]["counts"].encode("ascii")})
                    enc = mask_utils.encode(np.asfortranarray(union))
                    rle = {"size": enc["size"], "counts": enc["counts"].decode("ascii")}
                target.write_text(json.dumps({"image_size": [w, h], "model_identity": identity, "dynamic_rle": rle}))
                written += 1
        print(root, "масок записано:", written)


def body_from(source: Path, root: Path, arm: str, name: str) -> str:
    """Всё, кроме датасета и выхода, — из конфига RobotCar; модель при оценке берётся из чекпойнта."""

    lines = source.read_text().splitlines()
    out, skipping = [], False
    for line in lines:
        if line.startswith("dataset:"):
            skipping = True
            out += DATASET.replace("{root}", str(root)).rstrip("\n").splitlines()
            continue
        if skipping and (line.startswith("  ") or not line.strip()):
            continue
        skipping = False
        if line.startswith("output:"):
            out += ["output:", f"  root: {DATA / 'pmtransfer_runs'}", f"  experiment_name: {name}"]
            skipping = True
            continue
        out.append(line)
        if line.startswith("  dynamic_node_classes:"):
            out.append(f"  vocabulary_manifest: {VOCABULARY}")
            if arm == "shuf":
                out.append(f"  shuffle_text_seed: {SHUFFLE_SEED}")
    return "\n".join(out) + "\n"


def configs():
    n = 0
    for arm, root in ROOTS.items():
        source = C / ("final4_notext_f0_s42.yaml" if arm == "notext" else "final4_add_f0_s42.yaml")
        body = body_from(source, root, arm, f"pm-prepare-{arm}")
        needs = ["adapter: pervomay", "vocabulary_manifest:", "backend: mask_ioa", "threshold: 0.5",
                 "backend: char_ngram", "top_k: 100", str(root)]
        if arm != "notext":
            # Корень без текста на RobotCar готовился без фильтра мусора: строк в графе всё равно нет.
            needs += ["text_min_length: 3", "text_drop_numeric: true", "text_fusion: additive"]
        for need in needs:
            assert need in body, (arm, need)
        assert ("shuffle_text_seed" in body) == (arm == "shuf")
        assert ("use_text_nodes: false" in body) == (arm == "notext")
        (C / f"pm_prepare_{arm}.yaml").write_text(body)
        for weights in ("final4", "final4s"):
            for evaluation, (data_arm, weight_arm) in EVALS.items():
                if data_arm != arm:
                    continue
                for f in range(5):
                    name = f"pm-{weights}-{evaluation}-f{f}"
                    (C / f"pm_{weights}_{evaluation}_f{f}.yaml").write_text(body_from(source, root, arm, name))
                    checkpoint = KEEP / weights / f"{weights}-{weight_arm}-f{f}-s42" / "best_model.pth"
                    assert checkpoint.is_file(), checkpoint
                    n += 1
    print("конфигов оценок:", n)


if __name__ == "__main__":
    {"masks": masks, "configs": configs}[sys.argv[1]]()
