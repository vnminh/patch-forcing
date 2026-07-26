"""Patch Forcing transformer with spatial VITON control and garment cross-attention."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from einops import rearrange, repeat
from jaxtyping import Float

from jutils.nn.rope import make_axial_pos_2d
from jutils.nn.transformer import RMSNorm, TimestepEmbedder, TokenMerge2D, TokenSplitLast2D, TransformerLayer


class PatchForcingTransformerVTON(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        depth: int = 28,
        hidden_dim: int = 1152,
        head_dim: int = 72,
        mapping_dim: int = 384,
        mapping_depth: int = 2,
        patch_size: int = 2,
        garment_dim: int | None = None,
        compile: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        garment_dim = garment_dim or hidden_dim
        if hidden_dim % head_dim:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by head_dim={head_dim}.")
        self.latent_dim = latent_dim
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.gradient_checkpointing = gradient_checkpointing

        self.t_embedder = TimestepEmbedder(mapping_dim, mapping_depth, dim_mlp=3 * mapping_dim)
        self.merge = TokenMerge2D(latent_dim, hidden_dim, patch_size)
        self.spatial_gate = nn.Sequential(
            nn.Linear(hidden_dim + mapping_dim, mapping_dim),
            nn.SiLU(),
            nn.Linear(mapping_dim, 1),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=hidden_dim,
                    d_head=head_dim,
                    d_cond_norm=mapping_dim,
                    d_cross=garment_dim,
                    ff_expand=3,
                    rope_cls="jutils.nn.rope.AxialRoPE2D",
                    compile=compile,
                )
                for _ in range(depth)
            ]
        )
        self.split = TokenSplitLast2D(hidden_dim, latent_dim + 1, patch_size)

    def forward(
        self,
        x: Float[torch.Tensor, "b c h w"],
        t: Float[torch.Tensor, "b n"],
        spatial_tokens: Float[torch.Tensor, "b n d"],
        garment_tokens: Float[torch.Tensor, "b m d"],
        return_uncertainty: bool = False,
    ):
        b, c, h, w = x.shape
        if c != self.latent_dim:
            raise ValueError(f"Expected {self.latent_dim} latent channels, got {c}.")

        t_emb = self.t_embedder(t[..., None])
        pos = make_axial_pos_2d(h, w, device=x.device)
        pos = repeat(pos, "(h w) d -> b h w d", b=b, h=h, w=w)
        x = rearrange(x, "b c h w -> b h w c")
        x, pos = self.merge(x, pos)
        nh, nw, _ = x.shape[1:]
        x = rearrange(x, "b h w c -> b (h w) c")
        pos = rearrange(pos, "b h w d -> b (h w) d")
        n_tokens = x.shape[1]
        if t.shape != (b, n_tokens):
            raise ValueError(f"Expected t shape {(b, n_tokens)}, got {tuple(t.shape)}.")
        if spatial_tokens.shape != x.shape:
            raise ValueError(f"Expected spatial_tokens {tuple(x.shape)}, got {tuple(spatial_tokens.shape)}.")
        if garment_tokens.ndim != 3 or garment_tokens.shape[0] != b:
            raise ValueError(f"Invalid garment_tokens shape: {tuple(garment_tokens.shape)}.")

        # The encoder's final projection is zero-initialized; this residual starts as a no-op.
        gate = torch.sigmoid(self.spatial_gate(torch.cat([x, t_emb], dim=-1)))
        x = x + gate * spatial_tokens

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda tokens: block(tokens, pos=pos, cond_norm=t_emb, x_cross=garment_tokens),
                    x,
                    use_reentrant=False,
                )
            else:
                x = block(x, pos=pos, cond_norm=t_emb, x_cross=garment_tokens)
        x = rearrange(x, "b (h w) c -> b h w c", h=nh, w=nw)
        x = self.split(x)
        x = rearrange(x, "b h w c -> b c h w")
        velocity, logvar_theta = x[:, :-1], x[:, -1:]
        if return_uncertainty:
            return velocity, logvar_theta
        return velocity
