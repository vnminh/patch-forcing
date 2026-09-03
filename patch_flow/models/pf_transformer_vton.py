import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch.utils.checkpoint import checkpoint

from .pf_transformer import PatchForcingDiT, pf_modulate

GARMENT_SCALES = ("dino", "coarse", "middle", "detail")


class VTONPatchForcingBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        garment_scale=None,
        garment_attention_output_init_std=1e-3,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        self.garment_scale = garment_scale
        self.use_garment_cross_attention = garment_scale is not None
        if self.use_garment_cross_attention:
            if garment_attention_output_init_std <= 0:
                raise ValueError("garment_attention_output_init_std must be positive")
            self.garment_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.garment_cross_attention = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=0.0,
                batch_first=True,
            )
            # A small non-zero residual keeps the pretrained backbone nearly unchanged
            # while allowing Q/K/V and garment embedders to learn from the first step.
            nn.init.normal_(
                self.garment_cross_attention.out_proj.weight,
                std=float(garment_attention_output_init_std),
            )
            nn.init.zeros_(self.garment_cross_attention.out_proj.bias)

    def forward(self, x, c, garment_tokens=None, garment_padding_mask=None, edit_token_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(pf_modulate(self.norm1(x), shift_msa, scale_msa))
        if self.use_garment_cross_attention and garment_tokens is not None:
            cross, _ = self.garment_cross_attention(
                self.garment_norm(x),
                garment_tokens,
                garment_tokens,
                key_padding_mask=garment_padding_mask,
                need_weights=False,
            )
            if edit_token_mask is not None:
                cross = cross * edit_token_mask[..., None].to(cross.dtype)
            x = x + cross
        x = x + gate_mlp * self.mlp(pf_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class VTONPatchForcingDiT(PatchForcingDiT):
    """PFT-XL with person conditioning and multi-branch garment cross-attention.

    Garment appearance reaches the backbone through up to four branches:

    ``dino``    semantic DINOv2 patch features; establishes garment/body correspondence
    ``coarse``  garment VAE latent, embedded by a copy of the pretrained patch projection
    ``middle``  VAE encoder 1/4-resolution feature map
    ``detail``  VAE encoder 1/2-resolution feature map, the finest appearance carrier

    The DINO branch alone cannot transport logos, printed text, or colour blocks: its
    features are appearance-invariant by construction. The VAE branches live in the same
    representation space the pretrained denoiser already reads, so they can be copied.
    """

    def __init__(
        self,
        *args,
        person_condition_channels=5,
        use_dino_garment=True,
        garment_feature_dim=384,
        use_vae_garment=True,
        garment_middle_channels=None,
        garment_detail_channels=None,
        garment_scale_routes=None,
        garment_embed_gain=1.0,
        garment_attention_output_init_std=1e-3,
        cross_attention_every=4,
        gradient_checkpointing=False,
        pretrained_ckpt=None,
        pretrained_use_ema=True,
        **kwargs,
    ):
        kwargs["compile"] = False
        super().__init__(*args, **kwargs)
        if not self.predict_uncertainty:
            raise ValueError("VTONPatchForcingDiT requires predict_uncertainty=True")
        if person_condition_channels != 5:
            raise ValueError("person_condition_channels must be 5: four agnostic latent channels and one mask")
        if cross_attention_every < 1:
            raise ValueError("cross_attention_every must be positive")

        old_embedder = self.x_embedder
        input_size = old_embedder.img_size[0]
        patch_size = old_embedder.patch_size[0]
        self.state_channels = self.in_channels
        self.person_condition_channels = person_condition_channels
        self.use_dino_garment = bool(use_dino_garment)
        self.garment_feature_dim = int(garment_feature_dim)
        self.use_vae_garment = bool(use_vae_garment)
        self.garment_middle_channels = None if garment_middle_channels is None else int(garment_middle_channels)
        self.garment_detail_channels = None if garment_detail_channels is None else int(garment_detail_channels)
        self.garment_embed_gain = float(garment_embed_gain)
        self.garment_attention_output_init_std = float(garment_attention_output_init_std)
        self.cross_attention_every = cross_attention_every
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            self.state_channels + person_condition_channels,
            self.hidden_size,
            bias=True,
            strict_img_size=False,
        )

        self.garment_feature_norm = None
        self.garment_feature_proj = None
        self.garment_position_scale = None
        if self.use_dino_garment:
            self.garment_feature_norm = nn.LayerNorm(self.garment_feature_dim)
            self.garment_feature_proj = nn.Linear(self.garment_feature_dim, self.hidden_size, bias=True)
            # Unit gain avoids flattening the attention logits into a uniform average over
            # every garment token; the output projection separately uses a small residual init.
            nn.init.xavier_uniform_(self.garment_feature_proj.weight, gain=self.garment_embed_gain)
            nn.init.zeros_(self.garment_feature_proj.bias)
            # DINO supplies its own positional encoding, so the DiT grid embedding is optional.
            self.garment_position_scale = nn.Parameter(torch.zeros(()))

        self.garment_embedder = None
        if self.use_vae_garment:
            self.garment_embedder = PatchEmbed(
                input_size,
                patch_size,
                self.state_channels,
                self.hidden_size,
                bias=True,
                strict_img_size=False,
            )
        self.garment_middle_embedder = None
        self.garment_detail_embedder = None
        if (self.garment_middle_channels is None) != (self.garment_detail_channels is None):
            raise ValueError("Both garment_middle_channels and garment_detail_channels must be configured")
        self.use_multiscale_garment = self.garment_middle_channels is not None
        if self.use_multiscale_garment:
            self.garment_middle_embedder = nn.Conv2d(
                self.garment_middle_channels, self.hidden_size, kernel_size=4, stride=4
            )
            self.garment_detail_embedder = nn.Conv2d(
                self.garment_detail_channels, self.hidden_size, kernel_size=4, stride=4
            )
            for embedder in (self.garment_middle_embedder, self.garment_detail_embedder):
                nn.init.xavier_uniform_(embedder.weight, gain=self.garment_embed_gain)
                nn.init.zeros_(embedder.bias)

        with torch.no_grad():
            self.x_embedder.proj.weight.zero_()
            self.x_embedder.proj.weight[:, : self.state_channels].copy_(old_embedder.proj.weight)
            self.x_embedder.proj.bias.copy_(old_embedder.proj.bias)
            if self.garment_embedder is not None:
                self.garment_embedder.proj.weight.copy_(old_embedder.proj.weight)
                self.garment_embedder.proj.bias.copy_(old_embedder.proj.bias)

        old_blocks = self.blocks
        cross_attention_count = sum((index + 1) % cross_attention_every == 0 for index in range(len(old_blocks)))
        self.garment_scale_routes = self._resolve_routes(garment_scale_routes, cross_attention_count)
        self.blocks = nn.ModuleList()
        route_index = 0
        for index, old_block in enumerate(old_blocks):
            garment_scale = None
            if (index + 1) % cross_attention_every == 0:
                garment_scale = self.garment_scale_routes[route_index]
                route_index += 1
            block = VTONPatchForcingBlock(
                self.hidden_size,
                self.num_heads,
                garment_scale=garment_scale,
                garment_attention_output_init_std=self.garment_attention_output_init_std,
            )
            block.load_state_dict(old_block.state_dict(), strict=False)
            self.blocks.append(block)

        if pretrained_ckpt is not None:
            self.load_pretrained_checkpoint(pretrained_ckpt, use_ema=pretrained_use_ema)

    def _enabled_scales(self):
        enabled = []
        if self.use_dino_garment:
            enabled.append("dino")
        if self.use_vae_garment:
            enabled.append("coarse")
        if self.use_multiscale_garment:
            enabled.extend(("middle", "detail"))
        return enabled

    def _resolve_routes(self, garment_scale_routes, cross_attention_count):
        enabled = self._enabled_scales()
        if not enabled:
            raise ValueError("At least one garment conditioning branch must be enabled")
        if garment_scale_routes is None:
            garment_scale_routes = [enabled[0]] * cross_attention_count
        else:
            garment_scale_routes = list(garment_scale_routes)
        if len(garment_scale_routes) != cross_attention_count:
            raise ValueError(
                f"Expected {cross_attention_count} garment scale routes, got {len(garment_scale_routes)}"
            )
        unknown = set(garment_scale_routes) - set(GARMENT_SCALES)
        if unknown:
            raise ValueError(f"Garment scale routes must be one of {list(GARMENT_SCALES)}, got {sorted(unknown)}")
        disabled = set(garment_scale_routes) - set(enabled)
        if disabled:
            raise ValueError(f"Garment scale routes {sorted(disabled)} reference disabled branches")
        return tuple(garment_scale_routes)

    @staticmethod
    def _select_checkpoint_state(checkpoint, use_ema=True):
        state = checkpoint.get("state_dict", checkpoint)
        prefixes = ("ema_model.", "model.") if use_ema else ("model.", "ema_model.")
        for prefix in prefixes:
            selected = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
            if selected:
                return selected
        return state

    def load_pretrained_checkpoint(self, checkpoint_path, use_ema=True):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        return self.load_pretrained_state_dict(self._select_checkpoint_state(checkpoint, use_ema=use_ema))

    def load_pretrained_state_dict(self, pretrained_state):
        current = self.state_dict()
        loaded = set()
        for key, value in pretrained_state.items():
            if key in current and current[key].shape == value.shape:
                current[key] = value
                loaded.add(key)

        weight_key = "x_embedder.proj.weight"
        bias_key = "x_embedder.proj.bias"
        if weight_key in pretrained_state:
            base_weight = pretrained_state[weight_key]
            if base_weight.shape[1] != self.state_channels:
                raise ValueError(f"Expected {self.state_channels} pretrained input channels, got {base_weight.shape[1]}")
            current[weight_key].zero_()
            current[weight_key][:, : self.state_channels] = base_weight
            if self.garment_embedder is not None:
                current["garment_embedder.proj.weight"] = base_weight.clone()
            loaded.add(weight_key)
        if bias_key in pretrained_state:
            current[bias_key] = pretrained_state[bias_key]
            if self.garment_embedder is not None:
                current["garment_embedder.proj.bias"] = pretrained_state[bias_key].clone()
            loaded.add(bias_key)

        ignored = {key for key in pretrained_state if key not in loaded}
        if ignored:
            raise RuntimeError(f"Could not transfer pretrained PFT parameters: {sorted(ignored)}")
        self.load_state_dict(current, strict=True)
        return self

    def _token_mask(self, mask, spatial_size):
        if mask is None:
            return None
        mask = F.interpolate(mask.float(), size=spatial_size, mode="nearest")
        mask = F.max_pool2d(mask, kernel_size=self.patch_size, stride=self.patch_size)
        return mask.flatten(2).squeeze(1) > 0.5

    def _position_embedding(self, height, width, dtype, device):
        patch_height, patch_width = self.x_embedder.patch_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        return self._grid_position_embedding(grid_height, grid_width, dtype, device)

    def _grid_position_embedding(self, grid_height, grid_width, dtype, device):
        base_height, base_width = self.x_embedder.grid_size
        if (grid_height, grid_width) == (base_height, base_width):
            return self.pos_embed.to(device=device, dtype=dtype)
        position = self.pos_embed.reshape(1, base_height, base_width, self.hidden_size).permute(0, 3, 1, 2)
        position = F.interpolate(position.float(), size=(grid_height, grid_width), mode="bicubic", align_corners=False)
        return position.permute(0, 2, 3, 1).flatten(1, 2).to(device=device, dtype=dtype)

    def _unpatchify_rectangular(self, tokens, height, width):
        patch_height, patch_width = self.x_embedder.patch_size
        grid_height = height // patch_height
        grid_width = width // patch_width
        if tokens.shape[1] != grid_height * grid_width:
            raise ValueError(f"Expected {grid_height * grid_width} output tokens, got {tokens.shape[1]}")
        tokens = tokens.reshape(
            tokens.shape[0], grid_height, grid_width, patch_height, patch_width, self.out_channels
        )
        tokens = torch.einsum("nhwpqc->nchpwq", tokens)
        return tokens.reshape(
            tokens.shape[0], self.out_channels, grid_height * patch_height, grid_width * patch_width
        )

    def _garment_branches(self, garment_features, garment, garment_middle, garment_detail, x, position, height, width):
        tokens = {}
        grids = {}
        token_height = height // self.patch_size
        token_width = width // self.patch_size

        if garment_features is not None and self.use_dino_garment:
            if garment_features.ndim != 4 or garment_features.shape[1] != self.garment_feature_dim:
                raise ValueError(
                    f"Expected garment features (B,{self.garment_feature_dim},H,W), "
                    f"got {tuple(garment_features.shape)}"
                )
            features = F.interpolate(
                garment_features,
                size=(token_height, token_width),
                mode="bilinear",
                align_corners=False,
            )
            features = features.flatten(2).transpose(1, 2)
            features = self.garment_feature_norm(features)
            dino = self.garment_feature_proj(features).to(x.dtype)
            tokens["dino"] = dino + self.garment_position_scale.to(x.dtype) * position
            grids["dino"] = (token_height, token_width)

        if garment is not None and self.garment_embedder is not None:
            latent = F.interpolate(garment, size=(height, width), mode="bilinear", align_corners=False)
            tokens["coarse"] = self.garment_embedder(latent) + position
            grids["coarse"] = (token_height, token_width)

        for name, source, embedder in (
            ("middle", garment_middle, self.garment_middle_embedder),
            ("detail", garment_detail, self.garment_detail_embedder),
        ):
            if source is None:
                continue
            if embedder is None:
                raise ValueError(f"garment_{name}_channels must be configured to use the '{name}' branch")
            embedded = embedder(source)
            grid = (embedded.shape[-2], embedded.shape[-1])
            embedded = embedded.flatten(2).transpose(1, 2)
            tokens[name] = embedded.to(x.dtype) + self._grid_position_embedding(*grid, x.dtype, x.device)
            grids[name] = grid
        return tokens, grids

    def _garment_padding_masks(self, garment_mask, grids):
        padding = {}
        if garment_mask is None:
            return padding
        for name, (grid_height, grid_width) in grids.items():
            mask_size = (grid_height * self.patch_size, grid_width * self.patch_size)
            keep = self._token_mask(garment_mask, mask_size)
            if keep is None:
                continue
            empty = ~keep.any(dim=1)
            if empty.any():
                keep = keep.clone()
                keep[empty, 0] = True
            padding[name] = ~keep
        return padding

    def forward(
        self,
        x,
        t,
        y=None,
        person_agnostic=None,
        person_mask=None,
        edit_mask=None,
        garment_features=None,
        garment=None,
        garment_middle=None,
        garment_detail=None,
        garment_mask=None,
        return_uncertainty=False,
    ):
        batch, _, height, width = x.shape
        if person_agnostic is None:
            person_agnostic = torch.zeros_like(x)
        if person_mask is None:
            person_mask = torch.zeros((batch, 1, height, width), device=x.device, dtype=x.dtype)
        person_agnostic = F.interpolate(person_agnostic, size=(height, width), mode="bilinear", align_corners=False)
        person_mask = F.interpolate(person_mask.float(), size=(height, width), mode="area").to(x.dtype)
        if edit_mask is None:
            edit_mask = person_mask
        x = torch.cat((x, person_agnostic, person_mask), dim=1)
        position = self._position_embedding(height, width, x.dtype, x.device)
        x = self.x_embedder(x) + position

        if t.ndim != 2 or t.shape[1] != x.shape[1]:
            raise ValueError(f"Expected per-token timesteps {(batch, x.shape[1])}, got {tuple(t.shape)}")
        cond = self.t_embedder(t[..., None]).squeeze(1)
        if self.y_embedder is not None:
            if y is None:
                y = torch.full((batch,), self.y_embedder.num_classes, device=x.device, dtype=torch.long)
            cond = cond + self.y_embedder(y, self.training)[:, None, :]

        garment_tokens, garment_grids = self._garment_branches(
            garment_features, garment, garment_middle, garment_detail, x, position, height, width
        )
        garment_padding_masks = self._garment_padding_masks(garment_mask, garment_grids)

        edit_token_mask = self._token_mask(edit_mask, (height, width))
        for block in self.blocks:
            block_tokens = garment_tokens.get(block.garment_scale)
            block_padding_mask = garment_padding_masks.get(block.garment_scale)
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    cond,
                    block_tokens,
                    block_padding_mask,
                    edit_token_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, block_tokens, block_padding_mask, edit_token_mask)
        x = self.final_layer(x, cond)
        x = self._unpatchify_rectangular(x, height, width)
        logvar_theta = x[:, -1:, :, :]
        velocity = x[:, :-1, :, :]
        if return_uncertainty:
            return velocity, logvar_theta
        return velocity
