from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import torch

from tgseqloc.config import AppConfig


class FakeEncoder:
    """Small deterministic encoder used to keep tests offline."""

    embedding_dim = 4
    model_name = "fake-encoder"
    revision = "test"
    cache_identity = {"algorithm": "deterministic-test-v1"}

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            values = [float((sum(map(ord, text)) + offset * 17) % 31 + 1) for offset in range(4)]
            rows.append(values)
        return torch.tensor(rows, dtype=torch.float32) if rows else torch.empty((0, 4))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_v4rl_tree(
    root: Path,
    frame_count: int = 5,
    *,
    reference_sequence: str = "seq1",
    query_sequence: str = "seq2",
) -> tuple[dict[str, object], Path]:
    dataset = root / "V4RL"
    ocr_root = root / "ocr"
    graph_root = root / "graphs"
    prepared = root / "prepared"
    base_timestamp = 1_000_000_000_000_000_000
    gt_lines = []

    sequences = (reference_sequence, query_sequence)
    for sequence_offset, sequence in enumerate(sequences):
        frame_root = dataset / f"{sequence}_frames"
        frame_root.mkdir(parents=True)
        for index in range(frame_count):
            timestamp = base_timestamp + index * 10_000_000 + sequence_offset
            stem = f"{index:06d}_{timestamp:019d}"
            (frame_root / f"{stem}.png").write_bytes(f"{sequence}-{index}".encode())
            write_json(
                ocr_root / sequence / "chunk_0" / stem / "paddleocr_v5.json",
                {
                    "image_width": 100,
                    "image_height": 50,
                    "predictions": [
                        {"text": f"shop {index}", "confidence": 0.95, "bbox": [10, 5, 30, 15]},
                        {"text": "no text", "confidence": 1.0, "bbox": [0, 0, 5, 5]},
                        {"text": "weak", "confidence": 0.1, "bbox": [1, 1, 2, 2]},
                    ],
                },
            )
            write_json(
                graph_root / sequence / f"{sequence}_c000" / f"{index:06d}.json",
                {
                    "nodes": [
                        {
                            "id": "building",
                            "data": {
                                "class_name": "building",
                                "bbox_2d": {"xyxy": [0.05, 0.05, 0.8, 0.8]},
                            },
                        }
                    ],
                    "links": [
                        {"source": "building", "target": "building", "label": "self"},
                        {"source": "missing", "target": "building", "label": "dangling"},
                    ],
                },
            )
        if sequence == query_sequence:
            for index in range(frame_count):
                query_ts = base_timestamp + index * 10_000_000 + 1
                ref_ts = base_timestamp + index * 10_000_000
                gt_lines.append(f"{query_ts},{ref_ts},{ref_ts}")

    gt_path = dataset / "gt" / "gt_shop_street_1.json"
    gt_path.parent.mkdir(parents=True)
    gt_path.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
    settings: dict[str, object] = {
        "dataset": "v4rl",
        "dataset_root": dataset,
        "sequences": list(sequences),
        "ocr_root_template": str(ocr_root / "{sequence}" / "chunk_{chunk}"),
        "scene_graph_root_template": str(graph_root / "{sequence}"),
        "gt_path": gt_path,
        "output_root": prepared,
        "reference_sequence": reference_sequence,
        "query_sequence": query_sequence,
        "chunk_size": 200,
        "ocr_confidence_threshold": 0.5,
        "frame_batch_size": 3,
        "gt_tolerance_ns": 1_000,
        "test_ratio": 0.2,
        "validation_ratio": 0.25,
        "cache_policy": "reuse_if_compatible",
        "device": "cpu",
    }
    return settings, gt_path


def make_app_config(
    root: Path,
    frame_count: int = 10,
    *,
    reference_sequence: str = "seq1",
    query_sequence: str = "seq2",
) -> tuple[AppConfig, FakeEncoder]:
    settings, _ = make_v4rl_tree(
        root,
        frame_count,
        reference_sequence=reference_sequence,
        query_sequence=query_sequence,
    )
    config = AppConfig()
    config.dataset.root = Path(settings["dataset_root"])
    config.dataset.sequences = list(settings["sequences"])
    config.dataset.reference_sequence = str(settings["reference_sequence"])
    config.dataset.query_sequence = str(settings["query_sequence"])
    config.dataset.ocr_root_template = str(settings["ocr_root_template"])
    config.dataset.scene_graph_root_template = str(settings["scene_graph_root_template"])
    config.dataset.gt_path = Path(settings["gt_path"])
    config.dataset.prepared_root = Path(settings["output_root"])
    config.dataset.test_ratio = float(settings["test_ratio"])
    config.dataset.validation_ratio = float(settings["validation_ratio"])
    config.preprocess.ocr_confidence_threshold = 0.5
    config.preprocess.frame_batch_size = 4
    config.preprocess.gt_tolerance_ns = 1_000
    config.model.hidden_dim = 8
    config.model.n_layers = 1
    config.model.proj_dim = 4
    config.model.node_emb_dim = 4
    config.model.edge_emb_dim = 4
    config.model.text_emb_dim = 4
    config.model.dropout = 0.0
    config.model.heads = 1
    config.training.epochs = 1
    config.training.patience = 1
    config.training.batch_size = 2
    config.training.infer_batch_size = 8
    config.training.negatives_per_query = 1
    config.training.hard_search_depth = frame_count
    config.training.recall_values = [1]
    config.training.early_stopping_metric = "R@1"
    config.retrieval.recall_values = [1]
    config.runtime.device = "cpu"
    config.runtime.num_workers = 0
    config.output.root = root / "outputs"
    config.output.experiment_name = "e2e"
    return config.validate(), FakeEncoder()
