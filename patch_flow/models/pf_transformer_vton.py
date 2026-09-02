import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch.utils.checkpoint import checkpoint

from .pf_transformer import PatchForcingDiT, pf_modulate


class VTONPatchForcingBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_garment_cross_attention=False):
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
        self.use_garment_cross_attention = use_garment_cross_attention
        if use_garment_cross_attention:
            self.garment_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.garment_cross_attention = nn.MultiheadAttention(
                hidden_size,
                num_heads,
                dropout=0.0,
                batch_first=True,
            )
            nn.init.zeros_(self.garment_cross_attention.out_proj.weight)
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
    def __init__(
        self,
        *args,
        person_condition_channels=5,
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
        self.cross_attention_every = cross_attention_every
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.x_embedder = PatchEmbed(
            input_size,
            patch_size,
            self.state_channels + person_condition_channels,
            self.hidden_size,
            bias=True,
        )
        self.garment_embedder = PatchEmbed(
            input_size,
            patch_size,
            self.state_channels,
            self.hidden_size,
            bias=True,
        )
        with torch.no_grad():
            self.x_embedder.proj.weight.zero_()
            self.x_embedder.proj.weight[:, : self.state_channels].copy_(old_embedder.proj.weight)
            self.x_embedder.proj.bias.copy_(old_embedder.proj.bias)
            self.garment_embedder.proj.weight.copy_(old_embedder.proj.weight)
            self.garment_embedder.proj.bias.copy_(old_embedder.proj.bias)

        old_blocks = self.blocks
        self.blocks = nn.ModuleList()
        for index, old_block in enumerate(old_blocks):
            block = VTONPatchForcingBlock(
                self.hidden_size,
                self.num_heads,
                use_garment_cross_attention=(index + 1) % cross_attention_every == 0,
            )
            block.load_state_dict(old_block.state_dict(), strict=False)
            self.blocks.append(block)

        if pretrained_ckpt is not None:
            self.load_pretrained_checkpoint(pretrained_ckpt, use_ema=pretrained_use_ema)

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
            current["garment_embedder.proj.weight"] = base_weight.clone()
            loaded.add(weight_key)
        if bias_key in pretrained_state:
            current[bias_key] = pretrained_state[bias_key]
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

    def forward(
        self,
        x,
        t,
        y=None,
        person_agnostic=None,
        person_mask=None,
        garment=None,
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
        x = torch.cat((x, person_agnostic, person_mask), dim=1)
        x = self.x_embedder(x) + self.pos_embed

        if t.ndim != 2 or t.shape[1] != x.shape[1]:
            raise ValueError(f"Expected per-token timesteps {(batch, x.shape[1])}, got {tuple(t.shape)}")
        cond = self.t_embedder(t[..., None]).squeeze(1)
        if self.y_embedder is not None:
            if y is None:
                y = torch.full((batch,), self.y_embedder.num_classes, device=x.device, dtype=torch.long)
            cond = cond + self.y_embedder(y, self.training)[:, None, :]

        garment_tokens = None
        garment_padding_mask = None
        if garment is not None:
            garment = F.interpolate(garment, size=(height, width), mode="bilinear", align_corners=False)
            garment_tokens = self.garment_embedder(garment) + self.pos_embed
            garment_keep = self._token_mask(garment_mask, (height, width))
            if garment_keep is not None:
                empty = ~garment_keep.any(dim=1)
                if empty.any():
                    garment_keep = garment_keep.clone()
                    garment_keep[empty, 0] = True
                garment_padding_mask = ~garment_keep

        edit_token_mask = self._token_mask(person_mask, (height, width))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block,
                    x,
                    cond,
                    garment_tokens,
                    garment_padding_mask,
                    edit_token_mask,
                    use_reentrant=False,
                )
            else:
                x = block(x, cond, garment_tokens, garment_padding_mask, edit_token_mask)
        x = self.final_layer(x, cond)
        x = self.unpatchify(x)
        logvar_theta = x[:, -1:, :, :]
        velocity = x[:, :-1, :, :]
        if return_uncertainty:
            return velocity, logvar_theta
        return velocity
