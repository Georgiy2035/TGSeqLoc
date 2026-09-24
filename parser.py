"""Command-line arguments shared by train.py and eval.py.

Every experiment is described by a YAML file in ``configs/``; the arguments
below only say where the data lives on this machine and where to write.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tgseqloc.config import AppConfig, load_config


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", type=Path, required=True,
                        help="experiment configuration, e.g. configs/v4rl/set_mask2former.yaml")
    parser.add_argument("--datasets_folder", type=Path, default=Path("datasets"),
                        help="raw dataset files: V4RL ground truth, RobotCar INS logs, TGSeqLoc-Street poses")
    parser.add_argument("--annotations_folder", type=Path, default=Path("annotations"),
                        help="text and scene-graph annotations (downloaded or produced by annotate_*.py)")
    parser.add_argument("--prepared_folder", type=Path, default=Path("prepared"),
                        help="cache of prepared graphs, one sub-folder per configuration")
    parser.add_argument("--output_folder", type=Path, default=Path("runs"),
                        help="checkpoints, metrics and retrieved candidates")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


def load_experiment(args: argparse.Namespace) -> AppConfig:
    """The configuration named by ``--config`` with this machine's folders."""

    config = load_config(args.config)
    name = args.config.stem
    config.dataset.root = args.datasets_folder
    config.dataset.annotation_root = args.annotations_folder
    config.dataset.prepared_root = args.prepared_folder / f"{config.dataset.adapter}-{name}"
    config.output.root = args.output_folder
    config.output.experiment_name = name
    config.runtime.device = args.device
    config.runtime.num_workers = args.num_workers
    return config.validate()
