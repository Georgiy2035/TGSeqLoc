"""Таблица OCR-ablation: три распознавателя на одном во всём остальном пайплайне.

Помимо recall собирает то, без чего результат не интерпретируется: сколько
текста каждая модель вообще дала и насколько модели согласны между собой.
Если распознаватели читают почти одинаково, совпадение метрик говорит о
лёгкости датасета, а не об устойчивости метода, и это должно быть видно.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

RESULTS = Path.home() / "ocr_benchmark" / "results"
OUTPUTS = Path.home() / "TGSeqLoc" / "outputs"
SEEDS = range(42, 52)
ARMS = {
    "PaddleOCR v5": ("c10-real-s{seed}", "paddleocr_v5.json"),
    "Qwen3-VL 4B": ("abl-qwen-s{seed}", "qwen3_vl_4b.json"),
    "EasyOCR": ("abl-easyocr-s{seed}", "easyocr.json"),
}
BASELINE = "PaddleOCR v5"
T_QUANTILE = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 10: 2.262}


def mean_sd(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    return mean, math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def ci95(values: list[float]) -> float:
    """Половина доверительного интервала для среднего."""

    n = len(values)
    if n < 2:
        return float("nan")
    return T_QUANTILE.get(n, 2.262) * mean_sd(values)[1] / math.sqrt(n)


def recalls(pattern: str) -> dict[str, list]:
    rows: dict[str, list] = {"R@1": [], "R@5": [], "R@10": [], "epoch": []}
    for seed in SEEDS:
        path = OUTPUTS / pattern.format(seed=seed) / "final_metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text())
        for key in ("R@1", "R@5", "R@10"):
            rows[key].append(metrics[key])
        rows["epoch"].append(metrics.get("best_epoch"))
    return rows


def text_stats(sidecar: str) -> dict:
    """Плотность и характер текста, который дал этот распознаватель."""

    frames = boxes = 0
    strings: Counter = Counter()
    confidences: list[float] = []
    lengths: list[int] = []
    per_frame: dict[str, set[str]] = {}
    for sequence in ("seq1", "seq2"):
        root = RESULTS / f"v4rl_{sequence}"
        if not root.is_dir():
            continue
        for frame_dir in sorted(root.iterdir()):
            path = frame_dir / sidecar
            if not path.is_file():
                continue
            frames += 1
            texts: set[str] = set()
            for row in json.loads(path.read_text()).get("predictions", []):
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                boxes += 1
                strings[text.casefold()] += 1
                confidences.append(float(row.get("confidence", 0.0)))
                lengths.append(len(text))
                texts.add(text.casefold())
            per_frame[f"{sequence}/{frame_dir.name}"] = texts
    return {
        "frames": frames,
        "boxes_per_frame": boxes / frames if frames else 0.0,
        "unique": len(strings),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0.0,
        "per_frame": per_frame,
    }


def agreement(left: dict, right: dict) -> float:
    """Доля совпавших строк на общих кадрах, IoU множеств."""

    shared = set(left) & set(right)
    intersection = union = 0
    for key in shared:
        first, second = left[key], right[key]
        intersection += len(first & second)
        union += len(first | second)
    return intersection / union if union else float("nan")


def main() -> None:
    stats = {name: text_stats(sidecar) for name, (_, sidecar) in ARMS.items()}
    metrics = {name: recalls(pattern) for name, (pattern, _) in ARMS.items()}
    base = metrics[BASELINE]

    print("\n=== Recall (среднее ± 95 % ДИ по сидам) ===")
    header = "{:16} {:>3} {:>16} {:>16} {:>16}  {}".format(
        "распознаватель", "n", "R@1", "R@5", "R@10", "побед R@5"
    )
    print(header)
    for name, rows in metrics.items():
        n = len(rows["R@1"])
        if n == 0:
            print("{:16} {:>3}  нет прогонов".format(name, "-"))
            continue
        cells = [
            "{:6.2f} +- {:4.2f}".format(mean_sd(rows[k])[0], ci95(rows[k]))
            for k in ("R@1", "R@5", "R@10")
        ]
        if name == BASELINE or len(base["R@5"]) != n:
            wins = "-"
        else:
            won = sum(1 for a, b in zip(rows["R@5"], base["R@5"]) if a > b)
            wins = "{}/{}".format(won, n)
        print("{:16} {:>3} {} {}".format(name, n, " ".join(cells), wins))

    print("\n=== Текст, который дал распознаватель (V4RL, обе последовательности) ===")
    print("{:16} {:>8} {:>12} {:>11} {:>7} {:>7}".format(
        "распознаватель", "кадров", "боксов/кадр", "уникальных", "увер.", "длина"))
    for name, row in stats.items():
        print("{:16} {:>8} {:>12.2f} {:>11} {:>7.2f} {:>7.1f}".format(
            name, row["frames"], row["boxes_per_frame"],
            row["unique"], row["mean_confidence"], row["mean_length"]))

    print("\n=== Согласие распознавателей (IoU множеств строк на общих кадрах) ===")
    names = [name for name in ARMS if stats[name]["frames"]]
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            value = agreement(stats[left]["per_frame"], stats[right]["per_frame"])
            print("  {:16} против {:16} {:.3f}".format(left, right, value))

    print("\nНизкое согласие при совпавших метриках = устойчивость метода.")
    print("Высокое согласие = датасет лёгкий, вывод об устойчивости не следует.")


if __name__ == "__main__":
    main()
