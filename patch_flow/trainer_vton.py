import os

import torch
from torchvision.utils import save_image

from jutils import exists

from patch_flow.diagonal_gaussian import DiagonalGaussian
from patch_flow.dino_garment import TrainableDinoGarmentEncoder
from patch_flow.log_utils import log_images
from patch_flow.trainer import LatentFlowTrainer, un_normalize_ims
from patch_flow.vton_utils import compose_vton, masked_mean


class LatentVTONPatchForcingTrainer(LatentFlowTrainer):
    def __init__(
        self,
        *args,
        uncertainty_weight=0.01,
        outside_velocity_weight=0.01,
        garment_dropout_prob=0.1,
        garment_encoder_name="facebook/dinov2-small",
        garment_encoder_trainable_blocks=2,
        garment_encoder_input_size=(448, 336),
        garment_encoder_gradient_checkpointing=False,
        garment_encoder_lr=5e-6,
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
        self.garment_dropout_prob = float(garment_dropout_prob)
        self.garment_encoder_lr = float(garment_encoder_lr)
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
        person = batch.get("person", target)
        garment = batch["garment"]
        target_latent = batch.get("latent")
        if target_latent is None:
            target_latent = self.encode(target)
        dilation = self.flow.sample_training_dilation() if self.training else self.flow.mask_dilation_tokens
        edit_masks = self.flow.prepare_masks(
            batch["agnostic_mask"],
            target_latent.shape[-2:],
            target_latent.dtype,
            dilation_tokens=dilation,
        )
        remove_masks = self.flow.prepare_masks(
            batch["agnostic_mask"],
            target_latent.shape[-2:],
            target_latent.dtype,
            dilation_tokens=0,
        )
        person_agnostic_image = batch["person_agnostic"]
        agnostic_latent = self.encode(person_agnostic_image)
        person_condition = agnostic_latent * (1 - remove_masks.latent)
        person_context = agnostic_latent * (1 - edit_masks.latent)
        garment_features = self.garment_encoder(garment)
        return (
            target,
            person,
            person_agnostic_image,
            target_latent,
            person_context,
            person_condition,
            garment_features,
            edit_masks,
            remove_masks,
        )

    def _drop_garment(self, garment_features, garment_mask):
        if not self.training or self.garment_dropout_prob <= 0:
            return garment_features, garment_mask
        keep = (
            torch.rand(garment_features.shape[0], device=garment_features.device)
            >= self.garment_dropout_prob
        ).to(garment_features.dtype)
        keep_image = keep[:, None, None, None]
        garment_features = garment_features * keep_image
        if garment_mask is not None:
            garment_mask = garment_mask * keep_image
        return garment_features, garment_mask

    def forward(self, batch):
        (
            _,
            _,
            _,
            target,
            person_context,
            person_condition,
            garment_features,
            edit_masks,
            remove_masks,
        ) = self._encode_batch(batch)
        edit_mask = batch["agnostic_mask"].float()
        garment_mask = batch.get("garment_mask")
        garment_features, garment_mask = self._drop_garment(garment_features, garment_mask)
        xt, ut, timesteps, masks = self.flow.get_interpolants(
            target,
            person_context,
            edit_mask,
            masks=edit_masks,
        )
        label = self._label(batch, target.shape[0], target.device)
        velocity, logvar = self.model(
            x=xt,
            t=timesteps,
            y=label,
            person_agnostic=person_condition,
            person_mask=remove_masks.condition,
            edit_mask=edit_masks.condition,
            garment_mask=garment_mask,
            garment_features=garment_features,
            return_uncertainty=True,
        )

        flow_loss = masked_mean((velocity - ut).square(), edit_masks.latent)
        outside = 1 - edit_masks.latent
        outside_loss = masked_mean(velocity.square(), outside)
        distribution = DiagonalGaussian(mean=velocity.detach(), logvar=logvar)
        uncertainty_loss = masked_mean(distribution.nll(ut), edit_masks.latent)
        loss = flow_loss + self.outside_velocity_weight * outside_loss + self.uncertainty_weight * uncertainty_loss
        return loss, {
            "flow_loss": flow_loss,
            "outside_velocity_loss": outside_loss,
            "sigma_loss": uncertainty_loss,
            "editable_fraction": edit_masks.latent.mean(),
            "outside_fraction": outside.mean(),
            "zero_time_fraction": ((timesteps == 0) & edit_masks.token).float().sum()
            / edit_masks.token.float().sum().clamp_min(1),
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        (
            target_image,
            person_image,
            agnostic_image,
            target,
            person_context,
            person_condition,
            garment_features,
            edit_masks,
            remove_masks,
        ) = self._encode_batch(batch)
        edit_mask = batch["agnostic_mask"].float()
        garment_mask = batch.get("garment_mask")
        label = self._label(batch, target.shape[0], target.device)
        generator = self.generator.manual_seed(batch_idx + self.global_rank * 16102024)
        noise = torch.randn(target.shape, generator=generator, dtype=target.dtype).to(target.device)
        sample_model = self.ema_model if exists(self.ema_model) else self.model
        samples = self.flow.generate(
            model=sample_model,
            x=noise,
            person_agnostic=person_context,
            person_condition=person_condition,
            person_condition_mask=remove_masks.condition,
            edit_mask=edit_mask,
            garment_mask=garment_mask,
            garment_features=garment_features,
            y=label,
            **self.sample_kwargs,
        )
        generated = self.decode(samples)
        expanded = edit_masks.latent
        expanded = torch.nn.functional.interpolate(expanded, size=target_image.shape[-2:], mode="nearest")
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
                "agnostic": un_normalize_ims(agnostic_image[:8]),
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
