"""Condition encoders for person-aligned and garment-reference VITON inputs."""

import torch
import torch.nn as nn
from einops import rearrange, repeat

from jutils.nn.rope import make_axial_pos_2d
from jutils.nn.transformer import RMSNorm, TokenMerge2D, TransformerLayer, zero_init


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class StrideEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: tuple[int, ...]):
        super().__init__()
        layers = []
        current = in_channels
        for out_channels in channels:
            layers.append(ConvNormAct(current, out_channels, stride=2))
            current = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VTONConditionEncoder(nn.Module):
    """Produces spatial person tokens and garment cross-attention tokens.

    The agnostic input is already VAE-encoded by the trainer. DensePose retains
    person-grid alignment; cloth and cloth-mask use a separate reference encoder.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 1152,
        patch_size: int = 2,
        pose_channels: int = 128,
        spatial_channels: int = 256,
        garment_channels: int = 512,
        garment_refiner_depth: int = 2,
        garment_head_dim: int = 96,
        compile: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        # 1024x768 -> 128x96, spatially aligned with the frozen VAE latent.
        self.pose_encoder = StrideEncoder(3, (64, 96, pose_channels))
        fused_channels = latent_dim + pose_channels

        self.spatial_fuser = nn.Sequential(
            ConvNormAct(fused_channels, spatial_channels),
            ConvNormAct(spatial_channels, spatial_channels),
        )
        self.spatial_merge = TokenMerge2D(spatial_channels, hidden_dim, patch_size)
        self.spatial_norm = RMSNorm(hidden_dim)
        # Starts as a no-op residual. The projection learns before upstream encoders receive gradients.
        self.spatial_out = zero_init(nn.Linear(hidden_dim, hidden_dim, bias=False))

        # 1024x768 -> 16x12 garment grid after six stride-2 stages.
        self.garment_encoder = StrideEncoder(4, (64, 128, 256, 384, garment_channels, garment_channels))
        self.garment_proj = nn.Linear(garment_channels, hidden_dim, bias=False)
        self.garment_refiner = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=hidden_dim,
                    d_head=garment_head_dim,
                    ff_expand=3,
                    rope_cls="jutils.nn.rope.AxialRoPE2D",
                    compile=compile,
                )
                for _ in range(garment_refiner_depth)
            ]
        )
        self.null_garment_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def _spatial_tokens(
        self,
        agnostic_latent: torch.Tensor,
        densepose: torch.Tensor,
    ) -> torch.Tensor:
        pose = self.pose_encoder(densepose)
        if pose.shape[-2:] != agnostic_latent.shape[-2:]:
            raise ValueError(
                "DensePose encoder output must align with agnostic latent; "
                f"got {pose.shape[-2:]} and {agnostic_latent.shape[-2:]}."
            )
        spatial = self.spatial_fuser(torch.cat([agnostic_latent, pose], dim=1))
        b, _, h, w = spatial.shape
        pos = make_axial_pos_2d(h, w, device=spatial.device)
        pos = repeat(pos, "(h w) d -> b h w d", b=b, h=h, w=w)
        spatial = rearrange(spatial, "b c h w -> b h w c")
        spatial, _ = self.spatial_merge(spatial, pos)
        spatial = rearrange(spatial, "b h w c -> b (h w) c")
        return self.spatial_out(self.spatial_norm(spatial))

    def _garment_tokens(self, cloth: torch.Tensor, cloth_mask: torch.Tensor) -> torch.Tensor:
        if cloth_mask.shape[1] != 1:
            raise ValueError(f"cloth_mask must have one channel, got {cloth_mask.shape}.")
        garment_input = torch.cat([cloth, cloth_mask], dim=1)
        features = self.garment_encoder(garment_input)
        b, _, h, w = features.shape
        pos = make_axial_pos_2d(h, w, device=features.device)
        pos = repeat(pos, "(h w) d -> b h w d", b=b, h=h, w=w)
        tokens = rearrange(features, "b c h w -> b (h w) c")
        tokens = self.garment_proj(tokens)
        pos = rearrange(pos, "b h w d -> b (h w) d")
        for block in self.garment_refiner:
            tokens = block(tokens, pos=pos)
        return tokens

    def forward(
        self,
        agnostic_latent: torch.Tensor,
        densepose: torch.Tensor,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "spatial_tokens": self._spatial_tokens(agnostic_latent, densepose),
            "garment_tokens": self._garment_tokens(cloth, cloth_mask),
        }

    def unconditional(self, conditions: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Return CFG null conditions with the same batch and sequence shapes."""
        garment = conditions["garment_tokens"]
        return {
            "spatial_tokens": torch.zeros_like(conditions["spatial_tokens"]),
            "garment_tokens": self.null_garment_token.to(garment.dtype).expand_as(garment),
        }
