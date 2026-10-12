"""К каким объектам привязан текст на RobotCar и что было на выбор."""
import json
from collections import Counter
from pathlib import Path
import torch
from PIL import Image, ImageDraw

H = Path.home()
ROOT = H / "tgseqloc_data/rcgf_real_qwen/robotcar"
DS = Path("/mnt/external_usb_hdd/6YL/Datasets/robotcar")
man = json.load(open(ROOT / "manifest.json"))

# словарь классов: ищем в манифесте отображение имя → индекс
vocab = None
cands = []
for f in sorted((ROOT / "mappings").glob("*.json")):
    try: cands.append((f.name, json.load(open(f))))
    except Exception: pass
for key, value in list(man.items()) + cands:
    if "class" not in str(key).lower() and "vocab" not in str(key).lower(): continue
    if isinstance(value, dict) and len(value) > 20 and all(isinstance(v, int) for v in value.values()):
        vocab = {v: k for k, v in value.items()}; print("словарь классов из", key); break
    if isinstance(value, list) and len(value) > 20 and all(isinstance(v, str) for v in value):
        vocab = dict(enumerate(value)); print("словарь классов из", key); break
if vocab is None:
    print("ключи манифеста:", [k for k in man])
name = lambda i: vocab.get(int(i), f"#{int(i)}") if vocab else f"#{int(i)}"

split = json.load(open(ROOT / "mappings/temporal_split.json"))
paths = (split["database_paths"] + split["query_paths"])[::2]
chosen, alternatives, single, big = Counter(), Counter(), Counter(), Counter()
multi_examples, n_links, n_multi = [], 0, 0
examples = []
for rel in paths:
    g = torch.load(ROOT / rel, weights_only=False)
    if not bool(g.is_text.any()):
        continue
    x, t = g.x, g.is_text
    boxes = torch.stack([x[:, 0] - x[:, 2] / 2, x[:, 1] - x[:, 3] / 2, x[:, 0] + x[:, 2] / 2, x[:, 1] + x[:, 3] / 2], 1)
    objs = (~t).nonzero().flatten().tolist()
    strings = list(g.text_strings)
    src, dst = g.edge_index
    te = g.is_text_edge.bool()
    for k, ti in enumerate(t.nonzero().flatten().tolist()):
        att = [int(d) for s_, d, e in zip(src.tolist(), dst.tolist(), te.tolist()) if e and s_ == ti and not bool(t[d])]
        if not att:
            continue
        o = att[0]; n_links += 1
        cname = name(g.node_class[o])
        chosen[cname] += 1
        area = float((boxes[o, 2] - boxes[o, 0]) * (boxes[o, 3] - boxes[o, 1]))
        if area > 0.5:
            big[cname] += 1
        cx, cy = float(x[ti, 0]), float(x[ti, 1])
        cont = [j for j in objs if boxes[j, 0] <= cx <= boxes[j, 2] and boxes[j, 1] <= cy <= boxes[j, 3]]
        if len(cont) >= 2:
            n_multi += 1
            others = sorted({name(g.node_class[j]) for j in cont if j != o})
            alternatives[(cname, tuple(others))] += 1
            if len(multi_examples) < 12:
                multi_examples.append((Path(rel).stem, strings[k] if k < len(strings) else "?", cname, others, round(area, 2)))
        if len(examples) < 4 and area > 0.3 and len(cont) >= 2:
            examples.append(rel)

print("\nсвязей текст→объект: %d, из них с несколькими контейнерами: %d (%.0f %%)" % (n_links, n_multi, 100 * n_multi / n_links))
print("\nк каким классам привязан текст (топ-15):")
for c, v in chosen.most_common(15):
    print("  %-22s %5d (%4.1f %%)   из них объект больше полкадра: %d" % (c, v, 100 * v / n_links, big[c]))
print("\nкогда контейнеров несколько — выбран → что ещё было (топ-12):")
for (c, others), v in alternatives.most_common(12):
    print("  %4d  выбран %-14s | ещё: %s" % (v, c, ", ".join(others)))
print("\nпримеры:")
for stem, s, c, others, area in multi_examples:
    print("  %s  «%s» → %s (площадь %.2f) | ещё: %s" % (stem, s, c, area, ", ".join(others)))
