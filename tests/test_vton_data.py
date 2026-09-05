import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from PIL import Image

from patch_flow.vton_data import VTONHDDataset


class VTONDataTests(unittest.TestCase):
    def test_stable_viton_shift_scale_is_independent_and_keeps_masks_aligned(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for subdir in ("image", "cloth", "agnostic-mask", "cloth-mask"):
                (root / "train" / subdir).mkdir(parents=True, exist_ok=True)
            (root / "train_pairs.txt").write_text("sample.jpg sample.jpg\n", encoding="utf-8")

            rgb = Image.new("RGB", (24, 32), "black")
            rgb.paste((255, 255, 255), (8, 8, 16, 24))
            mask = Image.new("L", (24, 32), 0)
            mask.paste(255, (8, 8, 16, 24))
            rgb.save(root / "train" / "image" / "sample.jpg")
            rgb.save(root / "train" / "cloth" / "sample.jpg")
            mask.save(root / "train" / "agnostic-mask" / "sample_mask.png")
            mask.save(root / "train" / "cloth-mask" / "sample_mask.png")

            dataset = VTONHDDataset(
                root,
                image_size=[32, 24],
                random_shift_scale=True,
                shift_scale_prob=1.0,
                shift_limit=0.2,
                scale_limit=0.2,
            )
            # Person moves right; garment moves left. Both keep scale=1.
            with patch("patch_flow.vton_data.random.random", side_effect=[0.0, 0.0]), patch(
                "patch_flow.vton_data.random.uniform",
                side_effect=[0.2, 0.0, 1.0, -0.2, 0.0, 1.0],
            ):
                sample = dataset[0]

            person_x = torch.where(sample["agnostic_mask"][0] > 0.5)[1].float().mean()
            garment_x = torch.where(sample["garment_mask"][0] > 0.5)[1].float().mean()
            self.assertGreater(person_x.item(), garment_x.item())

            # Masked foreground pixels stay bright: the RGB image and mask used
            # exactly the same affine parameters on each side.
            person_foreground = (sample["person"][0] > 0.0).float()
            garment_foreground = (sample["garment"][0] > 0.0).float()
            self.assertGreater((person_foreground * sample["agnostic_mask"][0]).mean().item(), 0.1)
            self.assertGreater((garment_foreground * sample["garment_mask"][0]).mean().item(), 0.1)

    def test_shift_scale_config_rejects_invalid_ranges(self):
        with TemporaryDirectory() as directory:
            pair_list = Path(directory) / "train_pairs.txt"
            pair_list.write_text("sample.jpg sample.jpg\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "shift_scale_prob"):
                VTONHDDataset(directory, pair_list=str(pair_list), shift_scale_prob=1.1)
            with self.assertRaisesRegex(ValueError, "scale_limit"):
                VTONHDDataset(directory, pair_list=str(pair_list), scale_limit=1.0)


if __name__ == "__main__":
    unittest.main()
