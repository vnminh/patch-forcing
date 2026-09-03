from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class VTONMasks:
    condition: torch.Tensor
    token: torch.Tensor
    latent: torch.Tensor


def prepare_vton_masks(mask, latent_size, patch_size=2):
    """Convert a supplied agnostic mask into the three VTON mask representations.

    The token grid is the finest granularity the sampler can address, so a token is
    editable when any source pixel inside it is masked. No further dilation is applied:
    growing the envelope past the token grid forces the model to re-synthesise identity
    evidence (jaw, neck, hair) that it could otherwise copy.
    """
    if mask.ndim == 3:
        mask = mask[:, None]
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"Expected mask shape (B,1,H,W), got {tuple(mask.shape)}")
    latent_height, latent_width = latent_size
    if latent_height % patch_size or latent_width % patch_size:
        raise ValueError("Latent dimensions must be divisible by the PFT patch size")

    condition = F.interpolate(mask.float(), size=latent_size, mode="area").clamp(0, 1)
    token_size = (latent_height // patch_size, latent_width // patch_size)
    token = F.adaptive_max_pool2d(mask.float(), token_size) > 0
    latent = token.repeat_interleave(patch_size, -2).repeat_interleave(patch_size, -1).float()
    expanded_soft = F.avg_pool2d(latent, kernel_size=3, stride=1, padding=1) * latent
    condition = torch.maximum(condition * latent, expanded_soft)
    return VTONMasks(condition=condition, token=token.flatten(2).squeeze(1), latent=latent)


def masked_mean(value, mask):
    mask = mask.to(value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    denominator = mask.sum() * (value.numel() / mask.numel())
    return (value * mask).sum() / denominator.clamp_min(1.0)


def compose_vton(generated, person, mask, feather_radius=8):
    mask = F.interpolate(mask.float(), size=generated.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)
    if feather_radius > 0:
        kernel = 2 * feather_radius + 1
        softened = F.avg_pool2d(mask, kernel_size=kernel, stride=1, padding=feather_radius)
        mask = softened * (mask > 0).to(softened.dtype)
    return generated * mask + person * (1 - mask)
