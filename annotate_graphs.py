"""Scene-graph annotation of a folder of frames (Fig. 4, scene-graph branch).

Objects come from a detector-based generator and relations from box geometry
(see tgseqloc/annotation/graphs.py for the rules). Grounding DINO is queried
with twenty static street classes, so its graphs contain no dynamic objects;
Mask2Former graphs drop dynamic COCO classes unless ``--keep_dynamic`` is given.

One node-link JSON file per frame is written to ``--output_folder/{frame_id}.json``.

    python annotate_graphs.py --images_folder datasets/v4rl/seq1_frames \\
        --output_folder annotations/v4rl/graphs/grounding_dino/seq1 --generator grounding_dino
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tgseqloc.annotation.graphs import GENERATORS, build_graph
from tgseqloc.annotation.ocr import load_oriented_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--generator", choices=sorted(GENERATORS), default="grounding_dino")
    parser.add_argument("--neighbours", type=int, default=1,
                        help="relations per object, to its nearest objects by box centre")
    parser.add_argument("--keep_dynamic", action="store_true",
                        help="Mask2Former only: keep objects of dynamic COCO classes")
    parser.add_argument("--stride", type=int, default=1, help="annotate every N-th image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    images = sorted(p for p in args.images_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:: max(1, args.stride)]
    args.output_folder.mkdir(parents=True, exist_ok=True)
    if args.keep_dynamic and args.generator != "mask2former":
        parser.error("--keep_dynamic applies to --generator mask2former only")
    options = {"keep_dynamic": True} if args.keep_dynamic else {}
    generator = GENERATORS[args.generator](device=args.device, **options)

    written = nodes = 0
    for path in images:
        target = args.output_folder / f"{path.stem}.json"
        if target.exists():
            continue
        objects, raw = generator.objects(load_oriented_image(path))
        meta = {"scene_name": path.stem, **generator.identity, "neighbours": args.neighbours}
        graph = build_graph(objects, meta, neighbours=args.neighbours, raw_labels=raw)
        target.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
        written += 1
        nodes += len(objects)
    print(json.dumps({"frames": len(images), "written": written,
                      "nodes_per_frame": round(nodes / max(1, written), 2)}))


if __name__ == "__main__":
    main()
