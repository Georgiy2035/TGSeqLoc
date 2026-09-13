"""Блочная кросс-валидация RobotCar: режем маршрут на блоки, раздаём по фолдам с выравниванием
плотности текста, между частями оставляем буфер. Эталон плотности — Qwen с фильтром, один для всех веток."""
import json, math, sys
from pathlib import Path
import torch

H = Path.home()
REF = H / "tgseqloc_data/rcgf_real_qwen/robotcar"      # эталон плотности текста
INS = "/mnt/external_usb_hdd/6YL/Datasets/robotcar/2015-09-02-10-37-32/gps/ins.csv"
RADIUS = 25.0
FOLDS = 5

sys.path.insert(0, str(H / "TGSeqLoc/src"))
from tgseqloc.data.robotcar import load_ins_track

split = json.load(open(REF / "mappings/temporal_split.json"))
track = load_ins_track(Path(INS))
stems = [Path(rel).stem for rel in split["query_paths"]]
positions, has_text = {}, {}
for i, (rel, stem) in enumerate(zip(split["query_paths"], stems)):
    p = track.position_at(float(stem))
    if p is None:
        continue
    positions[i] = p
    has_text[i] = bool(torch.load(REF / rel, weights_only=False).is_text.any())
positives = {int(k) for k, v in split["positives"].items() if v}
print("запросов с координатами %d, с текстом %d (%.1f %%), с позитивами %d"
      % (len(positions), sum(has_text.values()), 100 * sum(has_text.values()) / len(positions), len(positives)))


def blocks_of(length_m):
    ordered = sorted(positions)
    blocks, current, travelled, prev = [], [], 0.0, None
    for index in ordered:
        p = positions[index]
        if prev is not None:
            step = math.dist(p, prev)
            travelled += step if step < 50 else 0.0   # разрыв — не пройденный путь
        if travelled >= length_m and current:
            blocks.append(current); current = []; travelled = 0.0
        current.append(index); prev = p
    if current:
        blocks.append(current)
    return blocks


def deal(blocks):
    """Змейкой по убыванию плотности текста: суммы плотностей у фолдов сходятся."""
    density = [sum(has_text[i] for i in b) / len(b) for b in blocks]
    order = sorted(range(len(blocks)), key=lambda b: (-density[b], b))
    assignment = {}
    for place, b in enumerate(order):
        cycle, step = divmod(place, FOLDS)
        assignment[b] = step if cycle % 2 == 0 else FOLDS - 1 - step
    return assignment, density


def separate(block_indices, reference_points):
    kept, dropped = [], []
    for index in block_indices:
        p = positions[index]
        (dropped if any(math.dist(p, q) <= RADIUS for q in reference_points) else kept).append(index)
    return kept, dropped


def build(length_m, verbose=False):
    blocks = blocks_of(length_m)
    assignment, density = deal(blocks)
    folds = {}
    for fold in range(FOLDS):
        val_fold = (fold + 1) % FOLDS
        test_raw = [i for b, f in assignment.items() if f == fold for i in blocks[b]]
        val_raw = [i for b, f in assignment.items() if f == val_fold for i in blocks[b]]
        train = [i for b, f in assignment.items() if f not in (fold, val_fold) for i in blocks[b]]
        train_pts = [positions[i] for i in train]
        val_kept, val_dropped = separate(val_raw, train_pts)
        held = train_pts + [positions[i] for i in val_kept]
        test_kept, test_dropped = separate(test_raw, held)
        keep = lambda idx: sorted(i for i in idx if i in positives)
        folds[fold] = dict(train=keep(train), validation=keep(val_kept), test=keep(test_kept),
                           dropped_validation=len(val_dropped), dropped_test=len(test_dropped))
    if verbose:
        print("\n=== блок %d м: блоков %d ===" % (length_m, len(blocks)))
        for fold, f in folds.items():
            d = lambda k: 100 * sum(has_text[i] for i in f[k]) / max(1, len(f[k]))
            print("  фолд %d: train %4d (%4.1f %% текста) | val %3d (%4.1f %%) | test %3d (%4.1f %%) | "
                  "в буфер ушло val %3d, test %3d"
                  % (fold, len(f["train"]), d("train"), len(f["validation"]), d("validation"),
                     len(f["test"]), d("test"), f["dropped_validation"], f["dropped_test"]))
        tested = sorted(i for f in folds.values() for i in f["test"])
        print("  всего тестируется за 5 фолдов: %d запросов, из них с текстом %d (сейчас 532 и 88)"
              % (len(tested), sum(has_text[i] for i in tested)))
    return blocks, folds


for length in (150, 200, 300):
    build(length, verbose=True)

LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 200
blocks, folds = build(LENGTH)
out = {"radius_m": RADIUS, "folds": FOLDS, "block_length_m": LENGTH,
       "density_reference": "rcgf_real_qwen", "traversal": "2015-09-02-10-37-32",
       "assignment": {str(k): {part: [stems[i] for i in v] for part, v in f.items() if isinstance(v, list)}
                      for k, f in folds.items()}}
path = H / "tgseqloc_data/robotcar_folds.json"
json.dump(out, open(path, "w"), indent=1)
print("\nразметка записана: %s (блок %d м)" % (path, LENGTH))
