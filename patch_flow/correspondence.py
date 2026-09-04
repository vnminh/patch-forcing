"""CORAL-style correspondence supervision for the garment cross-attention.

A frozen DINOv3 teacher matches every editable person token to the garment token it
actually corresponds to, by cosine similarity of patch features. That match is treated as
ground truth for *where* the cross-attention should look, and two losses are applied
directly to the attention distribution:

``centre of mass``
    the attention barycentre over garment positions is pulled onto the matched garment
    position. This is what turns "attend to the garment" into "attend to *this part* of
    the garment", which is exactly the placement signal a flow loss on latents gives only
    indirectly.

``entropy``
    a diffuse distribution has the same barycentre as a sharp one centred at the same
    point, so the centre-of-mass term alone is satisfied by attending uniformly to a
    symmetric neighbourhood. The entropy term removes that degenerate solution.

The teacher is **not** a conditioning branch: no DINO feature ever reaches the network.
It runs under ``no_grad`` on the ground-truth person image, which exists only at training
time, and disappears entirely at inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def grid_coordinates(grid, device=None, dtype=torch.float32):
    """Normalised (u, v) centres of a row-major ``grid`` of tokens, shape (H*W, 2)."""
    height, width = int(grid[0]), int(grid[1])
    rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
    columns = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
    v, u = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((u.reshape(-1), v.reshape(-1)), dim=-1)


def mask_to_token_valid(mask, grid):
    """Max-pool a pixel mask onto ``grid``; a token is valid if any pixel inside it is set."""
    if mask.ndim == 3:
        mask = mask[:, None]
    valid = F.adaptive_max_pool2d(mask.float(), (int(grid[0]), int(grid[1]))).flatten(1) > 0.5
    empty = ~valid.any(dim=1)
    if empty.any():
        # An empty key set would make every similarity -inf; fall back to the full grid so
        # the sample degrades to "no usable target" via the confidence gate instead of NaN.
        valid = valid.clone()
        valid[empty] = True
    return valid


class DinoCorrespondenceTeacher(nn.Module):
    """Frozen DINOv3 patch-feature extractor used only to build correspondence targets."""

    def __init__(
        self,
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        input_size=None,
        garment_grid=None,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_name = str(model_name)
        self.input_size = None if input_size is None else (int(input_size[0]), int(input_size[1]))
        self.garment_grid = None if garment_grid is None else (int(garment_grid[0]), int(garment_grid[1]))
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.requires_grad_(False)
        self.model.eval()

        patch_size = getattr(self.model.config, "patch_size", None)
        if patch_size is None:
            raise ValueError(f"{self.model_name} does not expose a patch size; it cannot be used as a teacher")
        self.patch_size = int(patch_size)
        if self.input_size is not None and any(size % self.patch_size for size in self.input_size):
            raise ValueError(f"Teacher input size {self.input_size} must be divisible by patch size {self.patch_size}")
        self.feature_dim = int(self.model.config.hidden_size)
        self.register_buffer("image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        # Frozen teacher: never leaves eval, so dropout and stochastic depth stay off and
        # the target for a given (person, garment) pair is deterministic across steps.
        super().train(False)
        return self

    def resolve_input_size(self, images):
        """Teacher resolution for ``images``.

        Defaults to the incoming resolution rounded to the patch grid. That matters: the
        dataset letterboxes person and garment to a fixed aspect ratio, and squeezing them
        into a differently shaped teacher input would shear both images and skew every
        matched position.
        """
        if self.input_size is not None:
            return self.input_size
        patch = self.patch_size
        return tuple(max(patch, int(round(size / patch)) * patch) for size in images.shape[-2:])

    def _forward_features(self, images):
        size = self.resolve_input_size(images)
        pixels = F.interpolate(
            (images.float() + 1) / 2,
            size=size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.model(pixel_values=pixels).last_hidden_state
        height, width = size[0] // self.patch_size, size[1] // self.patch_size
        prefix = tokens.shape[1] - height * width
        if prefix < 0:
            raise RuntimeError(f"Expected at least {height * width} patch tokens, got {tokens.shape[1]}")
        # DINOv2 prepends one CLS token, DINOv3 prepends CLS plus register tokens; both are
        # positional-free summaries and carry no correspondence information.
        tokens = tokens[:, prefix:]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.feature_dim, height, width)

    @torch.no_grad()
    def features(self, images, grid=None):
        features = self._forward_features(images)
        if grid is not None and tuple(int(size) for size in grid) != features.shape[-2:]:
            features = F.interpolate(features, size=(int(grid[0]), int(grid[1])), mode="bilinear", align_corners=False)
        return features

    @torch.no_grad()
    def correspondence(
        self,
        person,
        garment,
        person_grid,
        garment_mask=None,
        person_valid=None,
        min_similarity=0.0,
        soft_target_temperature=0.0,
        mutual=False,
        weight_by_similarity=False,
    ):
        """Match editable person tokens to garment tokens and return their positions.

        Returns ``(target, weight, similarity)`` where ``target`` is (B, Np, 2) normalised
        (u, v) garment coordinates, ``weight`` is (B, Np) supervision strength, and
        ``similarity`` is (B, Np) the raw best cosine similarity.
        """
        person_features = self.features(person, person_grid)
        garment_features = self.features(garment, self.garment_grid)
        garment_grid = tuple(garment_features.shape[-2:])
        return correspondence_targets(
            person_features,
            garment_features,
            garment_grid=garment_grid,
            person_valid=person_valid,
            garment_valid=None if garment_mask is None else mask_to_token_valid(garment_mask, garment_grid),
            min_similarity=min_similarity,
            soft_target_temperature=soft_target_temperature,
            mutual=mutual,
            weight_by_similarity=weight_by_similarity,
        )


@torch.no_grad()
def correspondence_targets(
    person_features,
    garment_features,
    garment_grid,
    person_valid=None,
    garment_valid=None,
    min_similarity=0.0,
    soft_target_temperature=0.0,
    mutual=False,
    weight_by_similarity=False,
):
    """Cosine-similarity correspondence from person tokens to garment positions."""
    person = F.normalize(person_features.float().flatten(2).transpose(1, 2), dim=-1)
    garment = F.normalize(garment_features.float().flatten(2).transpose(1, 2), dim=-1)
    similarity = torch.einsum("bpc,bgc->bpg", person, garment)
    neginf = torch.finfo(similarity.dtype).min

    masked = similarity
    if garment_valid is not None:
        masked = masked.masked_fill(~garment_valid[:, None, :], neginf)
    best_similarity, best_index = masked.max(dim=-1)

    coordinates = grid_coordinates(garment_grid, device=similarity.device, dtype=similarity.dtype)
    if soft_target_temperature > 0:
        attention = torch.softmax(masked / float(soft_target_temperature), dim=-1)
        target = attention @ coordinates
    else:
        target = coordinates[best_index]

    valid = torch.ones_like(best_similarity, dtype=torch.bool) if person_valid is None else person_valid.clone()
    valid = valid & (best_similarity >= float(min_similarity))
    if mutual:
        # Cycle consistency: the garment token a person token chose must choose it back.
        backward = similarity
        if person_valid is not None:
            backward = backward.masked_fill(~person_valid[:, :, None], neginf)
        back_index = backward.argmax(dim=1)
        positions = torch.arange(similarity.shape[1], device=similarity.device)
        valid = valid & (back_index.gather(1, best_index) == positions[None, :])

    weight = valid.to(similarity.dtype)
    if weight_by_similarity:
        weight = weight * best_similarity.clamp_min(0)
    return target, weight, best_similarity


class CorrespondenceAttentionLoss(nn.Module):
    """Centre-of-mass and entropy losses on garment cross-attention maps."""

    def __init__(self, center_weight=1.0, entropy_weight=0.05, entropy_eps=1e-8):
        super().__init__()
        self.center_weight = float(center_weight)
        self.entropy_weight = float(entropy_weight)
        self.entropy_eps = float(entropy_eps)

    @property
    def enabled(self):
        return self.center_weight > 0 or self.entropy_weight > 0

    def forward(self, attention_maps, target, weight):
        """``attention_maps`` is the list returned by ``VTONPatchForcingDiT``."""
        device = target.device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        if not attention_maps:
            return zero, {}

        weight = weight.float()
        denominator = weight.sum().clamp_min(1e-6)
        center_total = zero
        entropy_total = zero
        metrics = {}
        for entry in attention_maps:
            attention = entry["weights"].float()
            if attention.shape[1] != target.shape[1]:
                raise ValueError(
                    f"Attention has {attention.shape[1]} queries but the correspondence target has "
                    f"{target.shape[1]}; the query grid must be the person token grid"
                )
            coordinates = grid_coordinates(entry["grid"], device=device, dtype=attention.dtype)
            if coordinates.shape[0] != attention.shape[-1]:
                raise ValueError(
                    f"Branch '{entry['scale']}' has {attention.shape[-1]} keys but grid {entry['grid']}"
                )
            center = attention @ coordinates
            distance = (center - target).square().sum(-1)
            center_loss = (distance * weight).sum() / denominator

            logs = torch.log(attention.clamp_min(self.entropy_eps))
            entropy = -(attention * logs).sum(-1)
            padding = entry.get("key_padding")
            keys = (
                torch.full((attention.shape[0], 1), float(attention.shape[-1]), device=device)
                if padding is None
                else (~padding).sum(-1, keepdim=True).float()
            )
            # Normalise by the entropy of the uniform distribution over usable keys so the
            # 3072-key detail branch and the 768-key coarse branch contribute comparably.
            entropy = entropy / torch.log(keys.clamp_min(2.0))
            entropy_loss = (entropy * weight).sum() / denominator

            center_total = center_total + center_loss
            entropy_total = entropy_total + entropy_loss
            prefix = f"correspondence/block_{entry['block']:02d}_{entry['scale']}"
            metrics[f"{prefix}/center"] = center_loss.detach()
            metrics[f"{prefix}/entropy"] = entropy_loss.detach()

        count = len(attention_maps)
        center_total = center_total / count
        entropy_total = entropy_total / count
        loss = self.center_weight * center_total + self.entropy_weight * entropy_total
        metrics["correspondence_center_loss"] = center_total.detach()
        metrics["correspondence_entropy"] = entropy_total.detach()
        return loss, metrics
