"""Paired VITON-HD dataset with synchronized person and garment transforms."""

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


def _resolve_image(directory: Path, name: str) -> Path:
    """Resolve a VITON filename while tolerating JPG/PNG preprocessing outputs."""
    candidate = directory / name
    if candidate.exists():
        return candidate

    stem = Path(name).stem
    for suffix in (".jpg", ".png", ".jpeg"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve '{name}' in '{directory}'.")


class VitonHDSynchronizedTransform:
    """Resize VITON tensors while preserving paired geometric alignment."""

    def __init__(self, height: int = 1024, width: int = 768, random_horizontal_flip: bool = False):
        self.size = (int(height), int(width))
        self.random_horizontal_flip = random_horizontal_flip

    @staticmethod
    def _rgb(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
        image = TF.resize(image.convert("RGB"), size, interpolation=InterpolationMode.BILINEAR, antialias=True)
        return TF.to_tensor(image).mul_(2.0).sub_(1.0)

    @staticmethod
    def _mask(image: Image.Image, size: Tuple[int, int]) -> torch.Tensor:
        image = TF.resize(image.convert("L"), size, interpolation=InterpolationMode.NEAREST)
        return (TF.to_tensor(image) > 0.5).float()

    def __call__(self, sample: Dict[str, Image.Image]) -> Dict[str, torch.Tensor]:
        out = {
            "image": self._rgb(sample["image"], self.size),
            "agnostic": self._rgb(sample["agnostic"], self.size),
            "densepose": self._rgb(sample["densepose"], self.size),
            "cloth": self._rgb(sample["cloth"], self.size),
            "cloth_mask": self._mask(sample["cloth_mask"], self.size),
        }
        if self.random_horizontal_flip and torch.rand(()) < 0.5:
            # All tensors are image-aligned after resize, including the garment/mask pair.
            out = {key: TF.hflip(value) for key, value in out.items()}
        return out


class VitonHDDataset(Dataset):
    """VITON-HD train/test pairs with person-side and garment-side modality lookup."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        pairs_file: str | None = None,
        height: int = 1024,
        width: int = 768,
        random_horizontal_flip: bool = False,
    ):
        self.root = Path(root)
        self.split = split
        self.split_root = self.root / split
        if not self.split_root.is_dir():
            raise FileNotFoundError(f"VITON split directory does not exist: {self.split_root}")

        pair_path = Path(pairs_file) if pairs_file else self.root / f"{split}_pairs.txt"
        if not pair_path.is_file():
            raise FileNotFoundError(f"VITON pair file does not exist: {pair_path}")
        self.pairs = self._read_pairs(pair_path)
        self.transform = VitonHDSynchronizedTransform(height, width, random_horizontal_flip)

    @staticmethod
    def _read_pairs(path: Path) -> list[Tuple[str, str]]:
        pairs = []
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Expected '<person> <cloth>' pair in {path}, got: {line!r}")
            pairs.append((fields[0], fields[1]))
        if not pairs:
            raise ValueError(f"No pairs found in {path}")
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _load(self, directory: str, name: str) -> Image.Image:
        return Image.open(_resolve_image(self.split_root / directory, name))

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        person_name, cloth_name = self.pairs[index]
        sample = {
            "image": self._load("image", person_name),
            "agnostic": self._load("agnostic-v3.2", person_name),
            "densepose": self._load("image-densepose", person_name),
            "cloth": self._load("cloth", cloth_name),
            "cloth_mask": self._load("cloth-mask", cloth_name),
        }
        output = self.transform(sample)
        output["person_name"] = person_name
        output["cloth_name"] = cloth_name
        return output
