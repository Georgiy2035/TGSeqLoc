"""A1 — модель OCR на V4RL, эталонная конфигурация, новое разбиение (валидация — два блока), сид 42.

R@K — по test_retrievals.json (100 ближайших), p — знаковый перестановочный тест по блокам из 50 соседних
тестовых запросов, 95 % ДИ — бутстрэп по тем же блокам. Время OCR — медиана по всем кадрам V4RL из файлов
распознавания (у Qwen пакет 4 через vLLM: время пакета делится на его размер).
"""
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch

H = Path.home()
D = H / "tgseqloc_data"
OUT = H / "TGSeqLoc/outputs"
RUNS = D / "v4abl_runs"
ARMS = {  # имя в таблице: (прогон, корень, файл OCR)
    "без текста": ("notext", "abl_v4_notext", None),
    "PaddleOCR v5": ("add_paddle", "abl_v4_paddle", "paddleocr_v5"),
    "EasyOCR": ("add_easyocr", "abl_v4_easyocr", "easyocr"),
    "Qwen3-VL 4B (эталон)": ("add_qwen", "abl_v4_qwen", "qwen3_vl_4b"),
}
KS = (1, 5, 10, 25, 100)
PAIRS = (("PaddleOCR v5", "без текста"), ("EasyOCR", "без текста"), ("Qwen3-VL 4B (эталон)", "без текста"),
         ("Qwen3-VL 4B (эталон)", "PaddleOCR v5"), ("Qwen3-VL 4B (эталон)", "EasyOCR"), ("PaddleOCR v5", "EasyOCR"))

split = json.load(open(D / "abl_v4_qwen/v4rl/mappings/temporal_split.json"))
positives = {int(k): set(v) for k, v in split["positives"].items()}
test = split["test_query_indices"]
print("разбиение: train %d, val %d (блоки %s), test %d" % (len(split["train_query_indices"]),
      len(split["validation_query_indices"]), split["validation_blocks"], len(test)))
manifest = json.load(open(D / "abl_v4_qwen/v4rl/manifest.json"))
frames = manifest["source_identities"]["frames"]


def history(run):
    text = open(RUNS / f"v4abl_{run}_s42.json").read()
    return json.loads(text[text.index("{"):])


def ocr_ms(model):
    values = []
    for frame in frames:
        d = json.load(open(Path(frame["ocr"]["path"]).parent / f"{model}.json"))
        batch = int(d.get("batch_size") or 1)
        wall = d.get("batch_wall_time_ms") or d.get("inference_time_ms")
        if wall is not None:
            values.append(float(wall) / batch)
    return statistics.median(values)


hits, rows = {}, {}
for name, (run, root, model) in ARMS.items():
    r = json.load(open(OUT / f"v4abl-{run}-s42" / "test_retrievals.json"))
    assert r["query_indices"] == test
    hits[name] = {q: [int(any(c in positives[q] for c in near[:k])) for k in KS] for q, near in zip(r["query_indices"], r["nearest_database_indices"])}
    h = history(run)
    best = max(h["history"], key=lambda e: e["R@5"])
    text_frames = strings = 0.0
    if model:
        counts = []
        for q in test:
            g = torch.load(D / root / "v4rl" / split["query_paths"][q], map_location="cpu", weights_only=False)
            counts.append(int(g.is_text.sum()))
        text_frames = 100 * np.mean([c > 0 for c in counts]); strings = float(np.mean(counts))
    rows[name] = dict(recall=[100 * np.mean([hits[name][q][j] for q in test]) for j in range(len(KS))],
                      epochs=len(h["history"]), best=best["epoch"], val=best["R@5"], text=text_frames, strings=strings,
                      ms=ocr_ms(model) if model else None, val_curve=[e["R@5"] for e in h["history"]])

print("\n=== A1: модель OCR — V4RL, эталонная конфигурация, сид 42, тест %d запросов ===" % len(test))
print("%-22s %9s %8s %9s %7s %6s %7s  " % ("OCR", "текст, %", "строк/к", "мс/кадр", "эпох", "лучш.", "вал R@5") + "".join("%-8s" % f"R@{k}" for k in KS))
for name, row in rows.items():
    ms = "%9.0f" % row["ms"] if row["ms"] is not None else "%9s" % "—"
    text = "%9.1f %8.2f" % (row["text"], row["strings"]) if ARMS[name][2] else "%9s %8s" % ("—", "—")
    print("%-22s %s %s %7d %6d %7.1f  " % (name, text, ms, row["epochs"], row["best"], row["val"]) + "".join("%-8.2f" % v for v in row["recall"]))

blocks = [test[i:i + 50] for i in range(0, len(test), 50)]
rnd = random.Random(0)
print("\nпарные разницы (блоков %d по 50 запросов):" % len(blocks))
for x, y in PAIRS:
    parts = []
    for j, k in enumerate(KS[:3]):
        diff = [np.mean([hits[x][q][j] - hits[y][q][j] for q in b]) * 100 for b in blocks]
        weights = [len(b) for b in blocks]
        observed = np.average(diff, weights=weights)
        boot = sorted(np.average([diff[i] for i in s], weights=[weights[i] for i in s])
                      for s in ([rnd.randrange(len(blocks)) for _ in blocks] for _ in range(10000)))
        extreme = sum(abs(np.average([d * rnd.choice((1, -1)) for d in diff], weights=weights)) >= abs(observed) - 1e-12 for _ in range(20000))
        parts.append("R@%d %+.2f [%+.2f; %+.2f] p=%.3f" % (k, observed, boot[250], boot[9750], extreme / 20000))
    print("  %-40s %s" % (f"{x} − {y}", ";  ".join(parts)))

print("\nвалидация R@5 по эпохам:")
for name, row in rows.items():
    print("  %-22s %s" % (name, " ".join("%.1f" % v for v in row["val_curve"])))

repro, old = OUT / "repro-v4final-add-s42", OUT / "v4final-add-s42"
if (repro / "final_metrics.json").is_file():
    a, b = json.load(open(repro / "final_metrics.json")), json.load(open(old / "final_metrics.json"))
    same = all(a[k] == b[k] for k in ("R@1", "R@5", "R@10")) and \
        json.load(open(repro / "test_retrievals.json")) == json.load(open(old / "test_retrievals.json"))
    print("\nповтор v4final-add-s42 на текущем коде: %s (R@5 %.4f / %.4f)" % ("побитно совпал" if same else "НЕ СОВПАЛ", a["R@5"], b["R@5"]))
