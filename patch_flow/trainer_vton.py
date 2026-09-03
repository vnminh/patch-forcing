import os

import torch
from torchvision.utils import save_image

from jutils import exists

from patch_flow.diagonal_gaussian import DiagonalGaussian
from patch_flow.dino_garment import TrainableDinoGarmentEncoder
from patch_flow.log_utils import log_images
from patch_flow.trainer import LatentFlowTrainer, un_normalize_ims
from patch_flow.vae_features import encode_vae_pyramid
from patch_flow.vton_utils import compose_vton, masked_mean


class LatentVTONPatchForcingTrainer(LatentFlowTrainer):
    def __init__(
        self,
        *args,
        uncertainty_weight=0.01,
        outside_velocity_weight=0.01,
        detail_loss_weight=0.0,
        garment_dropout_prob=0.1,
        use_dino_garment=True,
        garment_encoder_name="facebook/dinov2-small",
        garment_encoder_trainable_blocks=0,
        garment_encoder_input_size=(448, 336),
        garment_encoder_gradient_checkpointing=False,
        garment_encoder_lr=2e-5,
        train_adapters_only=False,
        backbone_lr_multiplier=0.1,
        compute_validation_metrics=True,
        save_validation_previews=True,
        preview_every_n_validations=1,
        **kwargs,
    ):
        super().__init__(*args, enable_metrics=compute_validation_metrics, **kwargs)
        self.uncertainty_weight = float(uncertainty_weight)
        self.outside_velocity_weight = float(outside_velocity_weight)
        self.detail_loss_weight = float(detail_loss_weight)
        self.garment_dropout_prob = float(garment_dropout_prob)
        self.garment_encoder_lr = float(garment_encoder_lr)
        self.use_dino_garment = bool(use_dino_garment) and getattr(self.model, "use_dino_garment", True)
        self.garment_encoder = None
        if self.use_dino_garment:
            self.garment_encoder = TrainableDinoGarmentEncoder(
                model_name=garment_encoder_name,
                trainable_blocks=garment_encoder_trainable_blocks,
                input_size=garment_encoder_input_size,
                gradient_checkpointing=garment_encoder_gradient_checkpointing,
            )
            if self.garment_encoder.feature_dim != self.model.garment_feature_dim:
                raise ValueError(
                    f"DINO feature dimension {self.garment_encoder.feature_dim} does not match "
                    f"model garment_feature_dim {self.model.garment_feature_dim}"
                )
        self.use_vae_garment = bool(getattr(self.model, "use_vae_garment", False))
        self.use_multiscale_garment = bool(getattr(self.model, "use_multiscale_garment", False))
        if not (self.use_dino_garment or self.use_vae_garment or self.use_multiscale_garment):
            raise ValueError("At least one garment conditioning branch must be enabled")
        self.backbone_lr_multiplier = float(backbone_lr_multiplier)
        self.train_adapters_only = bool(train_adapters_only)
        self.compute_validation_metrics = bool(compute_validation_metrics)
        self.save_validation_previews = bool(save_validation_previews)
        self.preview_every_n_validations = int(preview_every_n_validations)
        if self.preview_every_n_validations < 1:
            raise ValueError("preview_every_n_validations must be positive")
        if train_adapters_only:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            for name, parameter in self.model.named_parameters():
                if "garment_" in name or name.startswith("x_embedder") or name.startswith("final_layer"):
                    parameter.requires_grad = True

    def configure_optimizers(self):
        adapter_parameters = []
        backbone_parameters = []
        garment_encoder_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("garment_encoder."):
                garment_encoder_parameters.append(parameter)
            elif "garment_" in name or ".x_embedder" in name:
                adapter_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)
        groups = []
        if adapter_parameters:
            groups.append({"params": adapter_parameters, "lr": self.lr})
        if backbone_parameters:
            groups.append({"params": backbone_parameters, "lr": self.lr * self.backbone_lr_multiplier})
        if garment_encoder_parameters:
            groups.append({"params": garment_encoder_parameters, "lr": self.garment_encoder_lr})
        optimizer = torch.optim.AdamW(groups, lr=self.lr, weight_decay=self.weight_decay)
        output = {"optimizer": optimizer}
        if exists(self.lr_scheduler_cfg):
            from jutils import load_partial_from_config

            output["lr_scheduler"] = load_partial_from_config(self.lr_scheduler_cfg)(optimizer=optimizer)
        return output

    @staticmethod
    def _gradient_norm(parameter, row_slice=None):
        if parameter is None or parameter.grad is None:
            return torch.zeros((), device=parameter.device if parameter is not None else "cpu")
        gradient = parameter.grad.detach()
        if row_slice is not None:
            gradient = gradient[row_slice]
        return torch.linalg.vector_norm(gradient.float())

    @torch.no_grad()
    def garment_gradient_norms(self):
        """Return pre-clipping gradient norms for every garment-conditioning path."""
        metrics = {}
        for block_index, block in enumerate(self.model.blocks, start=1):
            if not block.use_garment_cross_attention:
                continue
            attention = block.garment_cross_attention
            prefix = f"garment_grad/block_{block_index:02d}_{block.garment_scale}"
            metrics[f"{prefix}/out_proj"] = self._gradient_norm(attention.out_proj.weight)
            qkv_rows = attention.embed_dim
            metrics[f"{prefix}/q"] = self._gradient_norm(
                attention.in_proj_weight, slice(0, qkv_rows)
            )
            metrics[f"{prefix}/k"] = self._gradient_norm(
                attention.in_proj_weight, slice(qkv_rows, 2 * qkv_rows)
            )
            metrics[f"{prefix}/v"] = self._gradient_norm(
                attention.in_proj_weight, slice(2 * qkv_rows, 3 * qkv_rows)
            )

        embedders = {
            "dino": self.model.garment_feature_proj,
            "coarse": self.model.garment_embedder.proj if self.model.garment_embedder is not None else None,
            "middle": self.model.garment_middle_embedder,
            "detail": self.model.garment_detail_embedder,
        }
        for name, embedder in embedders.items():
            if embedder is not None:
                metrics[f"garment_grad/embedder_{name}"] = self._gradient_norm(embedder.weight)
        return metrics

    def _label(self, batch, batch_size, device):
        label = batch.get("label")
        if label is not None:
            return label.long().view(batch_size).to(device)
        return torch.full(
            (batch_size,),
            self.model.y_embedder.num_classes,
            device=device,
            dtype=torch.long,
        )

    def _encode_batch(self, batch):
        target = batch["image"]
        garment = batch["garment"]
        target_latent = batch.get("latent")
        if target_latent is None:
            target_latent = self.encode(target)
        # A single mask: the token grid is already the finest editable granularity, so the
        # generated region and the pixel-conditioned region coincide and no identity
        # evidence is thrown away.
        masks = self.flow.prepare_masks(
            batch["agnostic_mask"],
            target_latent.shape[-2:],
            target_latent.dtype,
        )
        agnostic_latent = self.encode(batch["person_agnostic"])
        person_context = agnostic_latent * (1 - masks.latent)

        garment_latent = batch.get("garment_latent")
        garment_middle = batch.get("garment_middle")
        garment_detail = batch.get("garment_detail")
        if self.use_multiscale_garment and (garment_middle is None or garment_detail is None):
            garment_latent, garment_middle, garment_detail = encode_vae_pyramid(self.first_stage, garment)
        elif self.use_vae_garment and garment_latent is None:
            garment_latent = self.encode(garment)
        if not self.use_vae_garment:
            garment_latent = None
        garment_features = self.garment_encoder(garment) if self.use_dino_garment else None
        return {
            "target_image": target,
            "person_image": batch.get("person", target),
            "agnostic_image": batch["person_agnostic"],
            "target": target_latent,
            "person_context": person_context,
            "masks": masks,
            "garment_features": garment_features,
            "garment": garment_latent,
            "garment_middle": garment_middle,
            "garment_detail": garment_detail,
        }

    def _garment_conditions(self, encoded):
        return {
            key: encoded[key]
            for key in ("garment_features", "garment", "garment_middle", "garment_detail")
        }

    def _drop_garment(self, conditions, garment_mask):
        if not self.training or self.garment_dropout_prob <= 0:
            return conditions, garment_mask
        reference = next((value for value in conditions.values() if value is not None), None)
        if reference is None:
            return conditions, garment_mask
        keep = (
            torch.rand(reference.shape[0], device=reference.device) >= self.garment_dropout_prob
        ).to(reference.dtype)
        keep_image = keep[:, None, None, None]
        conditions = {
            key: None if value is None else value * keep_image.to(value.dtype)
            for key, value in conditions.items()
        }
        if garment_mask is not None:
            garment_mask = garment_mask * keep_image.to(garment_mask.dtype)
        return conditions, garment_mask

    @staticmethod
    def _detail_loss(predicted, target, mask):
        """L1 on first spatial differences: penalises washed-out high-frequency structure
        (logo edges, printed text, colour-block seams) that a plain MSE tolerates."""
        horizontal_mask = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        vertical_mask = mask[:, :, 1:, :] * mask[:, :, :-1, :]
        horizontal = (predicted[:, :, :, 1:] - predicted[:, :, :, :-1]) - (
            target[:, :, :, 1:] - target[:, :, :, :-1]
        )
        vertical = (predicted[:, :, 1:, :] - predicted[:, :, :-1, :]) - (
            target[:, :, 1:, :] - target[:, :, :-1, :]
        )
        return masked_mean(horizontal.abs(), horizontal_mask) + masked_mean(vertical.abs(), vertical_mask)

    def forward(self, batch):
        encoded = self._encode_batch(batch)
        target = encoded["target"]
        masks = encoded["masks"]
        conditions, garment_mask = self._drop_garment(
            self._garment_conditions(encoded), batch.get("garment_mask")
        )
        xt, ut, timesteps, masks = self.flow.get_interpolants(
            target,
            encoded["person_context"],
            batch["agnostic_mask"].float(),
            masks=masks,
        )
        label = self._label(batch, target.shape[0], target.device)
        velocity, logvar = self.model(
            x=xt,
            t=timesteps,
            y=label,
            person_agnostic=encoded["person_context"],
            person_mask=masks.condition,
            edit_mask=masks.condition,
            garment_mask=garment_mask,
            return_uncertainty=True,
            **conditions,
        )

        flow_loss = masked_mean((velocity - ut).square(), masks.latent)
        outside = 1 - masks.latent
        outside_loss = masked_mean(velocity.square(), outside)
        distribution = DiagonalGaussian(mean=velocity.detach(), logvar=logvar)
        uncertainty_loss = masked_mean(distribution.nll(ut), masks.latent)
        loss = flow_loss + self.outside_velocity_weight * outside_loss + self.uncertainty_weight * uncertainty_loss
        metrics = {
            "flow_loss": flow_loss,
            "outside_velocity_loss": outside_loss,
            "sigma_loss": uncertainty_loss,
            "editable_fraction": masks.latent.mean(),
            "outside_fraction": outside.mean(),
            "zero_time_fraction": ((timesteps == 0) & masks.token).float().sum()
            / masks.token.float().sum().clamp_min(1),
            "high_time_fraction": ((timesteps > 0.9) & masks.token).float().sum()
            / masks.token.float().sum().clamp_min(1),
        }
        if self.detail_loss_weight > 0:
            time_latent = self.flow._tokens_to_latent(
                timesteps, target.shape[-2], target.shape[-1], target.dtype
            )
            predicted_clean = xt + (1 - time_latent) * velocity
            detail_loss = self._detail_loss(predicted_clean, target, masks.latent)
            loss = loss + self.detail_loss_weight * detail_loss
            metrics["detail_loss"] = detail_loss
        return loss, metrics

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        encoded = self._encode_batch(batch)
        target = encoded["target"]
        masks = encoded["masks"]
        target_image = encoded["target_image"]
        person_image = encoded["person_image"]
        label = self._label(batch, target.shape[0], target.device)
        generator = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(target.device)
        sample_model = self.ema_model if exists(self.ema_model) else self.model
        samples = self.flow.generate(
            model=sample_model,
            x=noise,
            person_agnostic=encoded["person_context"],
            person_condition=encoded["person_context"],
            person_condition_mask=masks.condition,
            edit_mask=batch["agnostic_mask"].float(),
            garment_mask=batch.get("garment_mask"),
            y=label,
            **self._garment_conditions(encoded),
            **self.sample_kwargs,
        )
        generated = self.decode(samples)
        expanded = torch.nn.functional.interpolate(
            masks.latent, size=target_image.shape[-2:], mode="nearest"
        )
        composed = compose_vton(generated, person_image, expanded)
        has_ground_truth = batch.get("has_ground_truth")
        if self.compute_validation_metrics:
            if has_ground_truth is None:
                self.metric_tracker(target_image, composed)
            elif has_ground_truth.any():
                self.metric_tracker(target_image[has_ground_truth], composed[has_ground_truth])
        if self.val_images is None:
            mask_preview = (expanded[:8].repeat(1, 3, 1, 1) * 255).clamp(0, 255).to(torch.uint8)
            self.val_images = {
                "target": un_normalize_ims(target_image[:8]),
                "person": un_normalize_ims(person_image[:8]),
                "agnostic": un_normalize_ims(encoded["agnostic_image"][:8]),
                "edit_mask": mask_preview,
                "garment": un_normalize_ims(batch["garment"][:8]),
                "tryon": un_normalize_ims(composed[:8]),
            }

    def on_validation_epoch_end(self):
        if self.val_images is not None:
            for key, images in self.val_images.items():
                log_images(self.logger, images, f"val/{key}/samples", stack="row", split=4, step=self.global_step)
            if self.save_validation_previews and (self.val_epochs + 1) % self.preview_every_n_validations == 0:
                self._save_validation_preview()
            self.val_images = None
        if self.compute_validation_metrics:
            metrics = self.metric_tracker.aggregate()
            for key, value in metrics.items():
                self.log(f"val/{key}", value, sync_dist=True)
            self.metric_tracker.reset()
        self.val_epochs += 1

    def _save_validation_preview(self):
        log_dir = getattr(self.logger, "log_dir", None)
        if not isinstance(log_dir, (str, os.PathLike)) or self.val_images is None:
            return
        keys = ("target", "person", "agnostic", "edit_mask", "garment", "tryon")
        rows = torch.stack([self.val_images[key] for key in keys], dim=1)
        rows = rows.flatten(0, 1).float().cpu() / 255
        preview_dir = os.path.join(log_dir, "previews")
        os.makedirs(preview_dir, exist_ok=True)
        step_path = os.path.join(preview_dir, f"step{self.global_step:06d}.png")
        latest_path = os.path.join(preview_dir, "latest.png")
        save_image(rows, step_path, nrow=len(keys), padding=4, pad_value=1)
        save_image(rows, latest_path, nrow=len(keys), padding=4, pad_value=1)
