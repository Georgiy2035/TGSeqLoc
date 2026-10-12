"""Многокамерный RobotCar: списки кадров mono-камер, маски динамики в формате пайплайна, конфиги пилота.

Сцена — у каждой камеры своя (графы боковых и задней будут от Олега), текст — PaddleOCR v5 на всех камерах,
фильтр динамики — YOLO11x-seg + IoA 0.5, как на фронте. Кадры других камер — дополнительные кадры тех же
проездов; позитив — любой кадр любой камеры в 25 м. Пилот: GAT 1 слой 256/256, hard mining, фолды 0 и 3, сид 42.

  python setup_multicam.py lists            — списки кадров mono-камер (разрыв с фронтальным кадром ≤ 0.5 с)
  python setup_multicam.py masks ROOT...    — маски YOLO mono-камер в stages/segmentation корней ROOT
  python setup_multicam.py configs          — конфиги пилота
"""
import csv, json, shutil, sys
from pathlib import Path

H = Path.home()
DATA = H / "tgseqloc_data"
CSV = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/frames_for_graphs.csv")
MONO_SEG = Path("/mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_seg")
CAMS = ("mono_left", "mono_right", "mono_rear")
ROLE = {"2015-08-12-15-04-18": "base", "2015-09-02-10-37-32": "query"}
# Где будут графы mono-камер: уточнить у Олега, формат — как у фронтальных Gemini ({sequence}/{timestamp}.json).
MONO_GRAPHS = str(DATA / "robotcar_scenegraphs_gemini_{camera}/{sequence}")


def lists():
    rows = list(csv.DictReader(open(CSV)))
    for cam in CAMS:
        stems = sorted({r[cam] for r in rows if float(r[cam + "_dt_s"]) <= 0.5})
        path = DATA / f"robotcar_frames_{cam}.txt"
        path.write_text(f"# RobotCar {cam}: ближайший по времени кадр к каждому фронтальному кадру графов, разрыв ≤ 0.5 с\n" + "\n".join(stems) + "\n")
        print(path, len(stems))


def masks(roots):
    from pycocotools import mask as mask_utils
    import numpy as np
    front_root = DATA / "rcgf_real_qwen/robotcar/stages/segmentation"
    identity = json.loads(next((front_root / "base").glob("*.json")).read_text())["model_identity"]
    for root in roots:
        seg = Path(root) / "robotcar/stages/segmentation"
        for role in ("base", "query"):
            (seg / role).mkdir(parents=True, exist_ok=True)
            for f in (front_root / role).glob("*.json"):
                target = seg / role / f.name
                if not target.exists():
                    shutil.copyfile(f, target)
        written = 0
        for traversal, role in ROLE.items():
            for cam in CAMS:
                for src in sorted((MONO_SEG / f"robotcar_seg_{traversal}_{cam}").glob("*/yolo11x_seg.json")):
                    target = seg / role / f"{cam}-{src.parent.name}.json"
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
        print(root, "масок mono записано:", written)


def configs():
    C = H / "TGSeqLoc/configs"
    n = 0
    for f in (0, 3):
        base = (C / f"f256_set_f{f}_s42.yaml").read_text().splitlines()
        def edit(lines, name, root, multicam, shuffled):
            out = []
            for l in lines:
                if l.startswith("  ocr_file_name:"):
                    l = "  ocr_file_name: paddleocr_v5.json"
                elif l.startswith("  prepared_root:"):
                    l = f"  prepared_root: {root}"
                elif l.startswith("  experiment_name:"):
                    l = f"  experiment_name: {name}"
                elif l.startswith("  epochs:"):
                    l = "  epochs: 15"
                elif l.startswith("  patience:"):
                    l = "  patience: 5"
                elif l.startswith("  shuffle_text_seed:") and not shuffled:
                    continue
                out.append(l)
                if l == "  proj_dim: 256":
                    out.append("  n_layers: 1")
                if l.startswith("  split_fold:") and multicam:
                    out.append("  primary_camera: stereo_centre")
                    out.append("  cameras:")
                    out.append("    stereo_centre: {}")
                    for cam in CAMS:
                        out.append(f"    {cam}:")
                        out.append(f"      ocr_root_template: /mnt/external_usb_hdd/6YL/sitnikov_sv/robotcar_mono_ocr/robotcar_{{traversal}}_{cam}")
                        out.append(f"      scene_graph_root_template: {MONO_GRAPHS.replace('{camera}', cam)}")
                        out.append(f"      image_path_template: /mnt/external_usb_hdd/6YL/Datasets/robotcar/{{traversal}}_rgb/{cam}")
                        out.append(f"      frame_list_path: {DATA}/robotcar_frames_{cam}.txt")
            body = "\n".join(out) + "\n"
            if shuffled and "shuffle_text_seed:" not in body:
                body = body.replace("preprocess:\n", "preprocess:\n  shuffle_text_seed: 20260911\n", 1)
            return body
        arms = {
            "front": (DATA / "rcgp_real_paddle", False, False),
            "multicam": (DATA / "rcgm_real_paddle", True, False),
            "multicam_shuf": (DATA / "rcgm_shuffled_paddle", True, True),
        }
        for arm, (root, multicam, shuffled) in arms.items():
            (C / f"mc_{arm}_f{f}_s42.yaml").write_text(edit(base, f"mc-{arm}-f{f}-s42", root, multicam, shuffled))
            n += 1
        # Без текста на тех же многокамерных кадрах: база и запросы шире, чем у фронтального варианта без текста.
        notext = (C / f"f256_notext_f{f}_s42.yaml").read_text().splitlines()
        (C / f"mc_multicam_notext_f{f}_s42.yaml").write_text(edit(notext, f"mc-multicam_notext-f{f}-s42", DATA / "rcgm_notext", True, False))
        n += 1
    print("конфигов:", n)


if __name__ == "__main__":
    {"lists": lambda: lists(), "masks": lambda: masks(sys.argv[2:]), "configs": lambda: configs()}[sys.argv[1]]()
