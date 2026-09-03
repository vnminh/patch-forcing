import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Model


class TrainableDinoGarmentEncoder(nn.Module):
    def __init__(
        self,
        model_name="facebook/dinov2-small",
        trainable_blocks=2,
        input_size=(448, 336),
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.input_size = tuple(int(value) for value in input_size)
        self.model = Dinov2Model.from_pretrained(self.model_name)
        layers = self.model.encoder.layer
        self.trainable_blocks = int(trainable_blocks)
        if not 0 <= self.trainable_blocks <= len(layers):
            raise ValueError(f"trainable_blocks must be in [0, {len(layers)}]")

        self.model.requires_grad_(False)
        self.full_finetune = self.trainable_blocks == len(layers)
        if self.full_finetune:
            self.model.requires_grad_(True)
        elif self.trainable_blocks > 0:
            for layer in layers[-self.trainable_blocks :]:
                layer.requires_grad_(True)
            self.model.layernorm.requires_grad_(True)
        if gradient_checkpointing and self.trainable_blocks > 0:
            self.model.gradient_checkpointing_enable()

        patch_size = int(self.model.config.patch_size)
        if any(size % patch_size for size in self.input_size):
            raise ValueError(f"DINO input dimensions must be divisible by patch size {patch_size}")
        self.feature_dim = int(self.model.config.hidden_size)
        self.grid_size = tuple(size // patch_size for size in self.input_size)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode=True):
        super().train(mode)
        if self.full_finetune:
            return self
        self.model.eval()
        if mode and self.trainable_blocks > 0:
            for layer in self.model.encoder.layer[-self.trainable_blocks :]:
                layer.train()
            self.model.layernorm.train()
        return self

    def forward(self, garment):
        pixels = F.interpolate(
            (garment.float() + 1) / 2,
            size=self.input_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pixels = (pixels - self.image_mean) / self.image_std
        tokens = self.model(pixel_values=pixels).last_hidden_state[:, 1:]
        height, width = self.grid_size
        if tokens.shape[1] != height * width:
            raise RuntimeError(f"Expected {height * width} DINO patch tokens, got {tokens.shape[1]}")
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.feature_dim, height, width)
