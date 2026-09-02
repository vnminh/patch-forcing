import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from tqdm import tqdm
from transformers import Dinov2Model

repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_dir)

from patch_flow.vton_data import _letterbox


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_image(path, height, width):
    image = Image.open(path).convert("RGB")
    image = _letterbox(image, (height, width), (127, 127, 127), Image.Resampling.BICUBIC)
    return TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)


def chunks(values, size):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def main(args):
    if args.height % 14 or args.width % 14:
        raise ValueError("DINO input height and width must be divisible by 14")
    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = Dinov2Model.from_pretrained(args.model).eval().requires_grad_(False).to(device)
    garment_dir = Path(args.dataset_root).resolve() / args.split / "cloth"
    output_dir = Path(args.output_dir).resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    garments = sorted(path for path in garment_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not args.overwrite:
        garments = [path for path in garments if not (output_dir / f"{path.stem}.pt").exists()]

    grid_height = args.height // model.config.patch_size
    grid_width = args.width // model.config.patch_size
    progress = tqdm(total=len(garments), desc=f"DINO {args.split}")
    for paths in chunks(garments, args.batch_size):
        images = torch.stack([load_image(path, args.height, args.width) for path in paths]).to(device)
        autocast = torch.autocast("cuda", dtype=dtype) if device.type == "cuda" else nullcontext()
        with torch.no_grad(), autocast:
            tokens = model(pixel_values=images).last_hidden_state[:, 1:]
        if tokens.shape[1] != grid_height * grid_width:
            raise RuntimeError(f"Expected {grid_height * grid_width} DINO patch tokens, got {tokens.shape[1]}")
        features = tokens.transpose(1, 2).reshape(
            tokens.shape[0], model.config.hidden_size, grid_height, grid_width
        )
        for path, feature in zip(paths, features):
            torch.save(feature.cpu().to(torch.float16), output_dir / f"{path.stem}.pt")
        progress.update(len(paths))
    progress.close()
    print(f"Saved DINO features to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--model", default="facebook/dinov2-small")
    parser.add_argument("--height", type=int, default=448)
    parser.add_argument("--width", type=int, default=336)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
