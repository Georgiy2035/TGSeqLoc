"""Финальный эксперимент: RobotCar, 4 камеры, графы Mask2Former, PaddleOCR, слагаемое против без текста.

Основа — конфиги neg1025 (GAT 1 слой 256/256, позитивы 10/25, до 30 эпох, patience 5, прогрев 10 эпох случайных
негативов, lr 3e-5, batch 8, dropout 0.1, triplet 0.3, char n-grams 64d, фильтры мусора и динамики).
Меняется: все 4 камеры как отдельные кадры, графы Mask2Former, PaddleOCR на всех камерах, свои подготовленные корни.
5 фолдов × сиды 42–44 × {слагаемое, без текста}.
"""
from pathlib import Path

H = Path.home()
C = H / "TGSeqLoc/configs"
DATA = H / "tgseqloc_data"
M2F = "/mnt/external_usb_hdd/6YL/Datasets/scene_graphs_robotcar_m2f/{traversal}_rgb/{camera}"
MONO_OCR = "/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/robotcar_{traversal}_{camera}"
IMAGES = "/mnt/external_usb_hdd/6YL/Datasets/robotcar/{traversal}_rgb/{camera}"
ROOTS = {"add": DATA / "rcg4_m2f_add_paddle", "notext": DATA / "rcg4_m2f_notext"}
CAMS = ("mono_left", "mono_right", "mono_rear")

n = 0
for arm in ("add", "notext"):
    for f in range(5):
        for s in (42, 43, 44):
            lines = (C / f"neg1025_{arm}_warm_f{f}_s{s}.yaml").read_text().splitlines()
            out = []
            for l in lines:
                if l.startswith("  ocr_file_name:"):
                    l = "  ocr_file_name: paddleocr_v5.json"
                elif l.startswith("  scene_graph_root_template:"):
                    l = f"  scene_graph_root_template: {M2F}"
                elif l.startswith("  image_path_template:"):
                    l = f"  image_path_template: {IMAGES}"
                elif l.startswith("  prepared_root:"):
                    l = f"  prepared_root: {ROOTS[arm]}"
                elif l.startswith("  experiment_name:"):
                    l = f"  experiment_name: final4-{arm}-f{f}-s{s}"
                out.append(l)
                if l.startswith("  split_fold:"):
                    out += ["  primary_camera: stereo_centre", "  camera_fold_max_gap_s: 0.3", "  cameras:", "    stereo_centre: {}"]
                    for cam in CAMS:
                        out += [f"    {cam}:",
                                f"      ocr_root_template: {MONO_OCR}",
                                f"      frame_list_path: {DATA}/robotcar_frames_{cam}.txt"]
            body = "\n".join(out) + "\n"
            for need in ("paddleocr_v5.json", "scene_graphs_robotcar_m2f", "n_layers: 1", "positive_max_distance_m: 10.0",
                         "miner_warmup_epochs: 10", "epochs: 30", "patience: 5", "cameras:", str(ROOTS[arm])):
                assert need in body, (arm, f, s, need)
            assert ("text_fusion: additive" in body) == (arm == "add")
            (C / f"final4_{arm}_f{f}_s{s}.yaml").write_text(body)
            n += 1
print("конфигов:", n)
