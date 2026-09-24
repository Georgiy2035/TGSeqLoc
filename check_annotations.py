"""Acceptance checks of text and scene-graph annotations (Fig. 4).

Checks coverage, absence of dynamic classes, box validity, dangling edges and
the binding between each file and its frame, for every sequence.

    python check_annotations.py --annotations_folder annotations \\
        --text v4rl/text/qwen3_vl_4b-yolo11x_seg --graphs v4rl/graphs/mask2former \\
        --sequences seq1 seq2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tgseqloc.annotation.checks import CheckReport, check_sequence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--annotations_folder", type=Path, default=Path("annotations"))
    parser.add_argument("--text", required=True, help="text annotation folder, relative to --annotations_folder")
    parser.add_argument("--graphs", required=True, help="graph annotation folder, relative to --annotations_folder")
    parser.add_argument("--sequences", nargs="+", required=True)
    args = parser.parse_args()

    report = CheckReport()
    for sequence in args.sequences:
        check_sequence(args.annotations_folder / args.text / sequence,
                       args.annotations_folder / args.graphs / sequence, report)
    print(json.dumps(report.summary(), indent=2, ensure_ascii=False))
    sys.exit(0 if report.accepted else 1)


if __name__ == "__main__":
    main()
