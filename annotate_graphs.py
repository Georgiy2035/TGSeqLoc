"""Scene-graph annotation of a folder of frames (Fig. 4, scene-graph branch).

Objects come from a detector-based generator and relations from box geometry
(see tgseqloc/annotation/graphs.py for the rules). Grounding DINO is queried
with twenty static street classes, so its graphs contain no dynamic objects;
Mask2Former graphs drop dynamic COCO classes unless ``--keep_dynamic`` is given.

The released V4RL graphs were produced from frames resized to 640x480
(bilinear) and stored as JPEG with quality 92; ``--resize 640 480`` applies
exactly that preprocessing in memory.

One node-link JSON file per frame is written to ``--output_folder/{frame_id}.json``.

    python annotate_graphs.py --images_folder datasets/v4rl/seq1_frames --resize 640 480 \\
        --output_folder annotations/v4rl/graphs/grounding_dino/seq1 --generator grounding_dino
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

from tgseqloc.annotation.graphs import GENERATORS, build_graph

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def load_frame(path: Path, resize: tuple[int, int] | None):
    """An RGB frame, optionally resized and passed through JPEG quality 92."""

    from PIL import Image

    image = Image.open(path).convert("RGB")
    if resize is None:
        return image
    buffer = io.BytesIO()
    image.resize(resize, Image.BILINEAR).save(buffer, "JPEG", quality=92)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_folder", type=Path, required=True)
    parser.add_argument("--output_folder", type=Path, required=True)
    parser.add_argument("--generator", choices=sorted(GENERATORS), default="grounding_dino")
    parser.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                        help="resize frames (bilinear, then JPEG q92) before detection")
    parser.add_argument("--neighbours", type=int, default=1,
                        help="relations per object, to its nearest objects by box centre")
    parser.add_argument("--keep_dynamic", action="store_true",
                        help="Mask2Former only: keep objects of dynamic COCO classes")
    parser.add_argument("--stride", type=int, default=1, help="annotate every N-th image")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.keep_dynamic and args.generator != "mask2former":
        parser.error("--keep_dynamic applies to --generator mask2former only")
    images = sorted(p for p in args.images_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = images[:: max(1, args.stride)]
    args.output_folder.mkdir(parents=True, exist_ok=True)
    options = {"keep_dynamic": True} if args.keep_dynamic else {}
    generator = GENERATORS[args.generator](device=args.device, **options)
    resize = tuple(args.resize) if args.resize else None

    written = nodes = 0
    for path in images:
        target = args.output_folder / f"{path.stem}.json"
        if target.exists():
            continue
        objects = generator.objects(load_frame(path, resize))
        graph = build_graph(objects, {"scene_name": path.stem, "generator": args.generator},
                            neighbours=args.neighbours)
        target.write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
        written += 1
        nodes += len(objects)
    print(json.dumps({"frames": len(images), "written": written,
                      "nodes_per_frame": round(nodes / max(1, written), 2)}))


if __name__ == "__main__":
    main()
