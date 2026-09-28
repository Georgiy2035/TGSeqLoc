"""EDTformer image descriptors of a folder of frames, for reranking (Sec. III-E).

The published EDTformer model (DINOv2 ViT-B/14 backbone, 4096-dimensional
descriptor) is loaded through torch.hub from its official repository. Frames
are read as RGB, normalized with the ImageNet statistics and resized to
322x322, as in the EDTformer evaluation code. The descriptors of all frames
are written to one ``.npz`` file with the arrays ``frame_ids`` (image file
names without extension) and ``descriptors``.

    python extract_image_descriptors.py --images_folder datasets/v4rl/seq1_frames \\
        --output image_descriptors/v4rl/seq1.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# Official repository at a fixed commit; its hubconf fetches the released weights (v1.0.0).
EDTFORMER_HUB = "Tong-Jin01/EDTformer:d0b85da3475e776b7baf87fbab5d5bfa6b50d723"


def load_edtformer(device: str):
    model = torch.hub.load(EDTFORMER_HUB, "EDTformer", trust_repo=True)
    return model.to(device).eval()


def main() -> None:
    from PIL import Image
    from torchvision.transforms import functional as F

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images_folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resize", type=int, nargs=2, default=[322, 322], metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    images = sorted(p for p in args.images_folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    model = load_edtformer(args.device)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    chunks = []
    with torch.inference_mode():
        for offset in range(0, len(images), args.batch_size):
            batch = []
            for path in images[offset: offset + args.batch_size]:
                tensor = F.normalize(F.to_tensor(Image.open(path).convert("RGB")), mean, std)
                batch.append(F.resize(tensor, args.resize))
            chunks.append(model(torch.stack(batch).to(args.device)).float().cpu().numpy())
    descriptors = np.concatenate(chunks) if chunks else np.zeros((0, 4096), dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, frame_ids=np.array([p.stem for p in images]), descriptors=descriptors)
    print(json.dumps({"frames": len(images), "dimension": int(descriptors.shape[1]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
