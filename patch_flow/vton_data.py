import os
import random

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


def _letterbox(image, size, fill, interpolation):
    width, height = image.size
    scale = min(size / width, size / height)
    resized = image.resize((round(width * scale), round(height * scale)), interpolation)
    canvas = Image.new(image.mode, (size, size), fill)
    left = (size - resized.width) // 2
    top = (size - resized.height) // 2
    canvas.paste(resized, (left, top))
    return canvas


def _resolve_file(directory, filename, mask=False):
    stem, _ = os.path.splitext(filename)
    candidates = [filename]
    if mask:
        candidates.extend((f"{stem}_mask.png", f"{stem}.png", f"{stem}.jpg"))
    for candidate in candidates:
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find '{filename}' in {directory}")


class VTONHDDataset(Dataset):
    def __init__(self, root, split="train", pair_list=None, image_size=256, random_flip=False, paired=True):
        self.root = os.path.abspath(root)
        self.split = split
        self.image_size = int(image_size)
        self.random_flip = bool(random_flip)
        self.paired = bool(paired)
        split_root = os.path.join(self.root, split)
        self.image_dir = os.path.join(split_root, "image")
        self.garment_dir = os.path.join(split_root, "cloth")
        self.agnostic_mask_dir = os.path.join(split_root, "agnostic-mask")
        self.garment_mask_dir = os.path.join(split_root, "cloth-mask")
        pair_list = pair_list or os.path.join(self.root, f"{split}_pairs.txt")
        if not os.path.isfile(pair_list):
            raise FileNotFoundError(f"Pair list not found: {pair_list}")
        with open(pair_list, "r", encoding="utf-8") as handle:
            self.pairs = [tuple(line.split()[:2]) for line in handle if line.strip()]
        if not self.pairs:
            raise ValueError(f"Pair list is empty: {pair_list}")

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, path):
        image = Image.open(path).convert("RGB")
        return _letterbox(image, self.image_size, (127, 127, 127), Image.Resampling.BICUBIC)

    def _load_mask(self, path):
        image = Image.open(path).convert("L")
        return _letterbox(image, self.image_size, 0, Image.Resampling.NEAREST)

    def __getitem__(self, index):
        person_name, garment_name = self.pairs[index]
        if self.paired:
            garment_name = person_name
        person = self._load_rgb(_resolve_file(self.image_dir, person_name))
        garment = self._load_rgb(_resolve_file(self.garment_dir, garment_name))
        agnostic_mask = self._load_mask(_resolve_file(self.agnostic_mask_dir, person_name, mask=True))
        garment_mask = self._load_mask(_resolve_file(self.garment_mask_dir, garment_name, mask=True))
        if self.random_flip and random.random() < 0.5:
            person = TF.hflip(person)
            garment = TF.hflip(garment)
            agnostic_mask = TF.hflip(agnostic_mask)
            garment_mask = TF.hflip(garment_mask)

        person = TF.to_tensor(person) * 2 - 1
        garment = TF.to_tensor(garment) * 2 - 1
        agnostic_mask = (TF.to_tensor(agnostic_mask) > 0.5).float()
        garment_mask = (TF.to_tensor(garment_mask) > 0.5).float()
        person_agnostic = person * (1 - agnostic_mask)
        return {
            "image": person.clone(),
            "person": person,
            "person_agnostic": person_agnostic,
            "garment": garment,
            "agnostic_mask": agnostic_mask,
            "garment_mask": garment_mask,
            "has_ground_truth": torch.tensor(person_name == garment_name),
            "person_name": person_name,
            "garment_name": garment_name,
        }


class DummyVTONDataset(Dataset):
    def __init__(self, num_samples=1024, image_size=256):
        self.num_samples = int(num_samples)
        self.image_size = int(image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        size = self.image_size
        person = torch.rand((3, size, size), generator=generator) * 2 - 1
        garment = torch.rand((3, size, size), generator=generator) * 2 - 1
        mask = torch.zeros((1, size, size))
        mask[:, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1
        garment_mask = mask.clone()
        return {
            "image": person.clone(),
            "person": person,
            "person_agnostic": person * (1 - mask),
            "garment": garment,
            "agnostic_mask": mask,
            "garment_mask": garment_mask,
        }
