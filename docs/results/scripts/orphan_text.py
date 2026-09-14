"""Доля «бесхозного» текста на RobotCar при разных определениях «объект найден»."""
import json, sys
from pathlib import Path
import torch

H = Path.home()
TEXT_CLASS_KEYS = ("building", "sign", "shop", "store", "fence", "wall", "bus", "pole", "lamp", "scaffold",
                   "billboard", "awning", "door", "window", "banner", "poster", "kiosk", "booth", "shelter",
                   "board", "notice", "hoarding", "facade", "canopy", "barrier", "bin", "box", "post")


def run(root_name):
    root = H / f"tgseqloc_data/{root_name}/robotcar"
    man = json.load(open(root / "manifest.json"))
    vocab = {v: k for k, v in man["node_class_to_idx"].items()}
    text_classes = {i for i, n in vocab.items() if any(k in n for k in TEXT_CLASS_KEYS) and "bus" != n}
    split = json.load(open(root / "mappings/temporal_split.json"))
    crit = {"центр внутри объекта (как сейчас)": 0, "≥50 % рамки текста внутри объекта": 0,
            "≥50 % внутри и объект ≤ полкадра": 0, "центр внутри объекта «текстового» класса": 0,
            "≥50 % внутри, ≤ полкадра, «текстовый» класс": 0}
    frames_any = {k: 0 for k in crit}
    n_text = n_frames = 0
    for rel in split["database_paths"] + split["query_paths"]:
        g = torch.load(root / rel, weights_only=False)
        t = g.is_text.bool()
        if not bool(t.any()):
            continue
        n_frames += 1
        x = g.x
        b = torch.stack([x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2, x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2], 1)
        objs = (~t).nonzero().flatten()
        ob = b[objs]; oarea = (ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1])
        ocls = g.node_class[objs]
        is_textual = torch.tensor([int(c) in text_classes for c in ocls], dtype=torch.bool)
        orphan_in_frame = {k: False for k in crit}
        for ti in t.nonzero().flatten():
            n_text += 1
            tb = b[ti]; cx, cy = x[ti, 0], x[ti, 1]
            tarea = max(float((tb[2] - tb[0]) * (tb[3] - tb[1])), 1e-9)
            inside = (ob[:, 0] <= cx) & (cx <= ob[:, 2]) & (ob[:, 1] <= cy) & (cy <= ob[:, 3])
            iw = (torch.minimum(ob[:, 2], tb[2]) - torch.maximum(ob[:, 0], tb[0])).clamp(min=0)
            ih = (torch.minimum(ob[:, 3], tb[3]) - torch.maximum(ob[:, 1], tb[1])).clamp(min=0)
            ioa = iw * ih / tarea
            found = {
                "центр внутри объекта (как сейчас)": bool(inside.any()),
                "≥50 % рамки текста внутри объекта": bool((ioa >= 0.5).any()),
                "≥50 % внутри и объект ≤ полкадра": bool(((ioa >= 0.5) & (oarea <= 0.5)).any()),
                "центр внутри объекта «текстового» класса": bool((inside & is_textual).any()),
                "≥50 % внутри, ≤ полкадра, «текстовый» класс": bool(((ioa >= 0.5) & (oarea <= 0.5) & is_textual).any()),
            }
            for k, ok in found.items():
                if not ok:
                    crit[k] += 1; orphan_in_frame[k] = True
        for k, v in orphan_in_frame.items():
            frames_any[k] += v
    print(f"\n=== {root_name}: надписей {n_text}, кадров с текстом {n_frames} ===")
    print("  %-46s %-22s %s" % ("критерий «объект найден»", "бесхозных надписей", "кадров с хоть одной бесхозной"))
    for k in crit:
        print("  %-46s %5d (%4.1f %%)        %4d (%4.1f %% кадров с текстом)"
              % (k, crit[k], 100 * crit[k] / n_text, frames_any[k], 100 * frames_any[k] / n_frames))
    return sorted(vocab[i] for i in text_classes)


classes = run("rcgf_real_qwen")
run("rcgf_real_paddle")
print("\n«текстовые» классы:", ", ".join(classes))
