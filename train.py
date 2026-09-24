"""Train TGSeqLoc on a dataset's annotations and evaluate the best checkpoint.

Graphs are prepared from the annotations, the encoder is trained with the
triplet loss (random negatives for the first ten epochs, hardest negatives
afterwards), the epoch with the best validation Recall@5 is kept, and it is
evaluated on the test queries.

    python train.py --config configs/v4rl/set_mask2former.yaml \\
        --datasets_folder datasets --annotations_folder annotations
"""

from __future__ import annotations

import argparse
import json

from parser import add_common_arguments, load_experiment
from tgseqloc.pipeline import PipelineRunner


def main() -> None:
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__.split("\n\n")[0]))
    parser.add_argument("--seed", type=int, default=None, help="overrides runtime.seed (42 in all configs)")
    args = parser.parse_args()

    config = load_experiment(args)
    if args.seed is not None:
        config.runtime.seed = args.seed
    result = PipelineRunner(config).run()
    print(json.dumps(result["training"].get("test_metrics", result["training"]), indent=2, default=str))


if __name__ == "__main__":
    main()
