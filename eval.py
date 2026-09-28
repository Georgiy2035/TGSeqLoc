"""Evaluate a trained TGSeqLoc model: Recall@1/5/10 on the test queries.

The annotations of the dataset named in the configuration are prepared into
graphs with the node-class and relation vocabularies stored in the checkpoint,
encoded by the model, and every test query is searched against the whole
database by exact cosine similarity (FAISS).

    python eval.py --config configs/v4rl/set_mask2former.yaml \\
        --resume weights/v4rl/set_mask2former.pth \\
        --datasets_folder datasets --annotations_folder annotations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from parser import add_common_arguments, load_experiment
from tgseqloc.evaluation.reranking import align_descriptors, load_image_descriptors
from tgseqloc.pipeline import PipelineRunner


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--resume", type=Path, required=True, help="model checkpoint (.pth)")
    parser.add_argument("--image_descriptors_folder", type=Path, default=None,
                        help="folder of {sequence}.npz from extract_image_descriptors.py; enables reranking")
    parser.add_argument("--rerank_num", type=int, default=1000,
                        help="candidates of the text-graph descriptor passed to reranking (K)")
    parser.add_argument("--sequence_frames", type=int, default=1,
                        help="query frames whose candidates are pooled for reranking (1: no sequence fusion)")
    args = parser.parse_args()

    config = load_experiment(args)
    # Graphs are built with the vocabularies the model was trained with.
    config.preprocess.vocabulary_manifest = str(args.resume)
    runner = PipelineRunner(config)
    runner.prepare()
    rerank = {}
    if args.image_descriptors_folder is not None:
        split = runner.split()
        sequences = sorted({str(path).split("/")[0] for path in split["database_paths"] + split["query_paths"]})
        images = load_image_descriptors(args.image_descriptors_folder, sequences)
        rerank = {
            "image_descriptors": (align_descriptors(split["database_paths"], images),
                                  align_descriptors(split["query_paths"], images)),
            "rerank_candidates": args.rerank_num,
            "sequence_frames": args.sequence_frames,
        }
    metrics = runner.transfer(args.resume, **rerank)
    recalls = {key: round(float(value), 2) for key, value in metrics.items() if key.startswith("R@")}
    report = {"config": str(args.config), "queries": metrics.get("num_evaluated_queries"), **recalls}
    if "reranked" in metrics:
        report["reranked"] = {key: (round(float(value), 2) if key.startswith("R@") else value)
                              for key, value in metrics["reranked"].items()}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
