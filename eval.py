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
from tgseqloc.pipeline import PipelineRunner


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--resume", type=Path, required=True, help="model checkpoint (.pth)")
    args = parser.parse_args()

    config = load_experiment(args)
    # Graphs are built with the vocabularies the model was trained with.
    config.preprocess.vocabulary_manifest = str(args.resume)
    runner = PipelineRunner(config)
    runner.prepare()
    metrics = runner.transfer(args.resume)
    recalls = {key: round(float(value), 2) for key, value in metrics.items() if key.startswith("R@")}
    print(json.dumps({"config": str(args.config), "queries": metrics.get("num_evaluated_queries"),
                      **recalls}, indent=2))


if __name__ == "__main__":
    main()
