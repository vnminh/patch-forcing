"""Trainer for VITON-HD Patch Forcing with spatial and garment conditions."""

import warnings
from copy import deepcopy
from typing import Union

import einops
import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule
from omegaconf import DictConfig
from torchmetrics import CatMetric

from jutils import exists, freeze, load_partial_from_config, tensor2im

from patch_flow.diagonal_gaussian import DiagonalGaussian
from patch_flow.integrators import EulerPF
from patch_flow.log_utils import log_image
from patch_flow.metrics import VitonMetricTracker
from patch_flow.trainer import instantiate_if_needed, update_ema


class PatchForcingVTONTrainer(LightningModule):
    def __init__(
        self,
        model: Union[dict, DictConfig, nn.Module],
        condition_encoder: Union[dict, DictConfig, nn.Module],
        first_stage: Union[dict, DictConfig, nn.Module],
        flow: Union[dict, DictConfig, object],
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        ema_rate: float = 0.9999,
        lr_scheduler_cfg: dict = None,
        uncertainty_weight: float = 0.01,
        condition_dropout_prob: float = 0.1,
        use_lpips: bool = False,
        sample_kwargs: dict = None,
    ):
        super().__init__()
        self.flow = instantiate_if_needed(flow)
        self.model = instantiate_if_needed(model)
        self.condition_encoder = instantiate_if_needed(condition_encoder)
        self.first_stage = instantiate_if_needed(first_stage)
        self.first_stage.eval().to(self.device)
        freeze(self.first_stage)

        self.ema_rate = ema_rate
        self.ema_model = None
        self.ema_condition_encoder = None
        if ema_rate > 0:
            if isinstance(model, nn.Module):
                warnings.warn("EMA model with deepcopy, might run into issues with compile.")
                self.ema_model = deepcopy(self.model)
                self.ema_condition_encoder = deepcopy(self.condition_encoder)
            else:
                self.ema_model = instantiate_if_needed(model)
                self.ema_model.load_state_dict(self.model.state_dict())
                self.ema_condition_encoder = instantiate_if_needed(condition_encoder)
                self.ema_condition_encoder.load_state_dict(self.condition_encoder.state_dict())
            freeze(self.ema_model)
            freeze(self.ema_condition_encoder)
            self.ema_model.eval()
            self.ema_condition_encoder.eval()
            update_ema(self.ema_model, self.model, decay=0)
            update_ema(self.ema_condition_encoder, self.condition_encoder, decay=0)

        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.uncertainty_weight = uncertainty_weight
        self.condition_dropout_prob = condition_dropout_prob
        self.sample_kwargs = sample_kwargs or {}
        self.generator = torch.Generator()
        self.metric_tracker = VitonMetricTracker(use_lpips=use_lpips).eval().to(self.device)
        self.val_losses = CatMetric().to(self.device)
        self.val_images = None
        self.val_epochs = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [parameter for parameter in self.parameters() if parameter.requires_grad],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        output = {"optimizer": optimizer}
        if exists(self.lr_scheduler_cfg):
            output["lr_scheduler"] = load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
        return output

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if exists(self._trainer) and exists(self.lr_scheduler_cfg):
            self.lr_schedulers().step()
        if exists(self.ema_model):
            update_ema(self.ema_model, self.model, decay=self.ema_rate)
            update_ema(self.ema_condition_encoder, self.condition_encoder, decay=self.ema_rate)

    @torch.no_grad()
    def encode(self, x):
        return self.first_stage.encode(x)

    @torch.no_grad()
    def decode(self, z):
        return self.first_stage.decode(z)

    def encode_conditions(self, batch, encoder=None, apply_dropout: bool = False):
        encoder = encoder or self.condition_encoder
        agnostic_latent = self.encode(batch["agnostic"])
        conditions = encoder(
            agnostic_latent=agnostic_latent,
            densepose=batch["densepose"],
            cloth=batch["cloth"],
            cloth_mask=batch["cloth_mask"],
        )
        if apply_dropout and self.condition_dropout_prob > 0:
            drop = torch.rand(conditions["spatial_tokens"].shape[0], device=agnostic_latent.device)
            drop = drop < self.condition_dropout_prob
            null = encoder.unconditional(conditions)
            conditions = {
                key: torch.where(drop[:, None, None], null[key], value) for key, value in conditions.items()
            }
        return conditions

    def forward(self, batch):
        target_latent = batch.get("latent")
        if target_latent is None:
            target_latent = self.encode(batch["image"])
        conditions = self.encode_conditions(batch, apply_dropout=True)
        xt, ut, t = self.flow.get_interpolants(x1=target_latent)
        vt, logvar_theta = self.model(x=xt, t=t, **conditions, return_uncertainty=True)
        flow_loss = (vt - ut).square().mean()
        sigma_theta = torch.exp(0.5 * logvar_theta)
        sigma_loss = DiagonalGaussian(mean=vt.detach(), std=sigma_theta).nll(ut).mean()
        loss = flow_loss + self.uncertainty_weight * sigma_loss
        return loss, {"flow_loss": flow_loss, "sigma_loss": sigma_loss}

    @torch.no_grad()
    def _sample(self, model, latent_noise, conditions):
        steps = int(self.sample_kwargs.get("num_steps", 50))
        sampler = EulerPF(patch_size=self.model.patch_size)
        timesteps = torch.linspace(0, 1, steps + 1, device=latent_noise.device)
        return sampler(model=model, x=latent_noise, timesteps=timesteps, progress=False, **conditions)

    def validation_step(self, batch, batch_idx):
        target_latent = batch.get("latent")
        if target_latent is None:
            target_latent = self.encode(batch["image"])
        conditions = self.encode_conditions(batch, encoder=self.condition_encoder, apply_dropout=False)
        sample_model = self.ema_model if exists(self.ema_model) else self.model
        sample_encoder = self.ema_condition_encoder if exists(self.ema_condition_encoder) else self.condition_encoder
        sample_conditions = self.encode_conditions(batch, encoder=sample_encoder, apply_dropout=False)

        generator = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        noise = torch.randn(target_latent.shape, generator=generator, dtype=target_latent.dtype, device=target_latent.device)
        _, val_loss_per_segment = self.flow.validation_losses(
            model=sample_model, x1=target_latent, x0=noise, **sample_conditions
        )
        self.val_losses.update(val_loss_per_segment.unsqueeze(0))

        samples = self.decode(self._sample(sample_model, noise, sample_conditions))
        self.metric_tracker(batch["image"], samples)

        if self.val_images is None:
            mask = batch["cloth_mask"].repeat(1, 3, 1, 1).mul(2).sub(1)
            views = [batch["image"], batch["agnostic"], batch["densepose"], batch["cloth"], mask, samples]
            grid = torch.cat(views, dim=2)
            grid = tensor2im(grid.float())
            grid = einops.rearrange(grid[:8], "b h w c -> h (b w) c")
            self.val_images = {"target_agnostic_densepose_cloth_mask_generated": grid}

    def on_validation_epoch_end(self):
        if self.val_images is not None:
            for key, image in self.val_images.items():
                log_image(self.logger, image, f"val/{key}", channel_last=True, step=self.global_step)
        self.val_images = None
        metrics = self.metric_tracker.aggregate()
        for key, value in metrics.items():
            self.log(f"val/{key}", value, sync_dist=True)
        self.metric_tracker.reset()

        if len(self.val_losses.value) > 0:
            self.log("val/loss", self.val_losses.compute().mean(), sync_dist=True)
            self.val_losses.reset()
        self.val_epochs += 1
        self.print(f"Val epoch {self.val_epochs:,} | Optimizer step {self.global_step:,}")
