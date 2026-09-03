import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from patch_flow.flow_vton import VTONPatchFlowForcing
from patch_flow.dino_garment import TrainableDinoGarmentEncoder
from patch_flow.models.pf_transformer import PatchForcingDiT
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.trainer_vton import LatentVTONPatchForcingTrainer
from patch_flow.vton_utils import compose_vton, prepare_vton_masks
from patch_flow.vton_data import VTONHDDataset
from train import repeat_dataloader


class FakeDinoOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class FakeDino(torch.nn.Module):
    """Minimal Dinov2Model stand-in: 12 blocks, patch size 14, and a real forward."""

    def __init__(self, hidden=4, patch_size=14):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(hidden, hidden) for _ in range(12)])
        self.embeddings = torch.nn.Linear(hidden, hidden)
        self.layernorm = torch.nn.LayerNorm(hidden)
        self.patch_embed = torch.nn.Conv2d(3, hidden, kernel_size=patch_size, stride=patch_size)
        self.config = SimpleNamespace(patch_size=patch_size, hidden_size=hidden)

    def forward(self, pixel_values=None):
        tokens = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        for layer in self.encoder.layer:
            tokens = layer(tokens)
        tokens = self.layernorm(tokens)
        # Prepend a CLS token; the encoder strips it with [:, 1:].
        return FakeDinoOutput(torch.cat((tokens[:, :1], tokens), dim=1))


def fake_dino_encoder(**kwargs):
    with patch("patch_flow.dino_garment.Dinov2Model.from_pretrained", return_value=FakeDino()):
        return TrainableDinoGarmentEncoder(**kwargs)


class ConstantVelocityModel(torch.nn.Module):
    def forward(self, x, t, return_uncertainty=False, **kwargs):
        velocity = torch.ones_like(x)
        if return_uncertainty:
            return velocity, torch.zeros_like(x[:, :1])
        return velocity


class RecordingVelocityModel(ConstantVelocityModel):
    def forward(self, x, t, person_agnostic=None, person_mask=None, edit_mask=None, **kwargs):
        self.person_condition = person_agnostic.detach().clone()
        self.person_mask = person_mask.detach().clone()
        self.edit_mask = edit_mask.detach().clone()
        return super().forward(x, t, **kwargs)


class VTONTests(unittest.TestCase):
    def test_dino_is_frozen_by_default(self):
        encoder = fake_dino_encoder()
        self.assertTrue(encoder.frozen)
        self.assertEqual(encoder.trainable_blocks, 0)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        # Must stay in eval even when the trainer switches to train mode.
        encoder.train()
        self.assertFalse(encoder.model.training)

    def test_frozen_dino_builds_no_autograd_graph(self):
        garment = torch.randn(1, 3, 448, 336, requires_grad=True)
        frozen = fake_dino_encoder(trainable_blocks=0).train()
        features = frozen(garment)
        self.assertEqual(tuple(features.shape[1:]), (4, 32, 24))
        self.assertIsNone(features.grad_fn)
        self.assertFalse(features.requires_grad)
        # Contrast: unfreezing restores the graph, so the no_grad path is what freezes it.
        unfrozen = fake_dino_encoder(trainable_blocks=2).train()
        self.assertTrue(unfrozen(garment).requires_grad)

    def test_only_requested_dino_tail_is_trainable(self):
        encoder = fake_dino_encoder(trainable_blocks=2)
        frozen = encoder.model.encoder.layer[:10]
        trainable = encoder.model.encoder.layer[10:]
        self.assertTrue(all(not parameter.requires_grad for layer in frozen for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for layer in trainable for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.model.layernorm.parameters()))

    def test_full_dino_finetuning_includes_embeddings(self):
        encoder = fake_dino_encoder(trainable_blocks=12)
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.model.parameters()))

    def test_preview_sample_is_moved_to_front(self):
        with TemporaryDirectory() as directory:
            pair_list = Path(directory) / "test_pairs.txt"
            pair_list.write_text("first.jpg first.jpg\ndetail.jpg detail.jpg\n", encoding="utf-8")
            dataset = VTONHDDataset(
                directory,
                split="test",
                pair_list=str(pair_list),
                preview_sample_id="detail.jpg",
            )
            self.assertEqual(dataset.pairs[0], ("detail.jpg", "detail.jpg"))

    def test_finite_dataloader_repeats_until_step_limit(self):
        batches = repeat_dataloader([0, 1, 2, 3], max_epochs=-1)
        self.assertEqual([next(batches) for _ in range(10)], [0, 1, 2, 3, 0, 1, 2, 3, 0, 1])

    def test_zero_initialized_model_matches_pft(self):
        torch.manual_seed(7)
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=2,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            compile=False,
        )
        base = PatchForcingDiT(**kwargs).eval()
        vton = VTONPatchForcingDiT(**kwargs, cross_attention_every=1).eval()
        vton.load_pretrained_state_dict(base.state_dict())
        x = torch.randn(2, 4, 8, 8)
        t = torch.rand(2, 16)
        y = torch.randint(0, 10, (2,))
        zeros = torch.zeros_like(x)
        mask = torch.zeros(2, 1, 8, 8)
        with torch.no_grad():
            expected = base(x, t, y)
            actual = vton(
                x,
                t,
                y,
                person_agnostic=zeros,
                person_mask=mask,
            )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_new_conditioning_paths_receive_gradients(self):
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            compile=False,
        )
        model = VTONPatchForcingDiT(
            **kwargs,
            garment_feature_dim=32,
            cross_attention_every=1,
            gradient_checkpointing=True,
        ).train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        x = torch.randn(1, 4, 8, 8)
        mask = torch.ones(1, 1, 8, 8)
        output = model(
            x,
            torch.rand(1, 16),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(x),
            person_mask=mask,
            garment_features=torch.randn(1, 32, 5, 4),
            garment_mask=mask,
        )
        output.square().mean().backward()
        self.assertGreater(model.x_embedder.proj.weight.grad[:, 4:].abs().sum().item(), 0)
        cross = model.blocks[0].garment_cross_attention.out_proj.weight.grad
        self.assertGreater(cross.abs().sum().item(), 0)

    def test_model_supports_rectangular_latents(self):
        model = VTONPatchForcingDiT(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            garment_feature_dim=32,
            cross_attention_every=1,
            compile=False,
        ).eval()
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        with torch.no_grad():
            velocity, logvar = model(
                latent,
                torch.rand(1, 12),
                torch.zeros(1, dtype=torch.long),
                person_agnostic=torch.randn_like(latent),
                person_mask=mask,
                edit_mask=mask,
                garment_features=torch.randn(1, 32, 5, 4),
                garment_mask=mask,
                return_uncertainty=True,
            )
        self.assertEqual(velocity.shape, latent.shape)
        self.assertEqual(logvar.shape, (1, 1, 8, 6))

    @staticmethod
    def _multiscale_model(**overrides):
        kwargs = dict(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=4,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            garment_feature_dim=32,
            garment_middle_channels=16,
            garment_detail_channels=8,
            garment_scale_routes=["dino", "coarse", "middle", "detail"],
            cross_attention_every=1,
            compile=False,
        )
        kwargs.update(overrides)
        return VTONPatchForcingDiT(**kwargs)

    def test_every_garment_branch_receives_gradients(self):
        model = self._multiscale_model().train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        output = model(
            latent,
            torch.rand(1, 12),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment_features=torch.randn(1, 32, 5, 4),
            garment=torch.randn(1, 4, 8, 6),
            garment_middle=torch.randn(1, 16, 16, 12),
            garment_detail=torch.randn(1, 8, 32, 24),
            garment_mask=mask,
        )
        output.square().mean().backward()
        for name, parameter in (
            ("dino", model.garment_feature_proj.weight),
            ("coarse", model.garment_embedder.proj.weight),
            ("middle", model.garment_middle_embedder.weight),
            ("detail", model.garment_detail_embedder.weight),
        ):
            self.assertIsNotNone(parameter.grad, f"{name} branch received no gradient")
            self.assertGreater(parameter.grad.abs().sum().item(), 0, f"{name} branch gradient is zero")

    def test_garment_gradient_metrics_cover_attention_and_embedders(self):
        model = self._multiscale_model().train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        output = model(
            latent,
            torch.rand(1, 12),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment_features=torch.randn(1, 32, 5, 4),
            garment=torch.randn(1, 4, 8, 6),
            garment_middle=torch.randn(1, 16, 16, 12),
            garment_detail=torch.randn(1, 8, 32, 24),
            garment_mask=mask,
        )
        output.square().mean().backward()

        trainer = object.__new__(LatentVTONPatchForcingTrainer)
        trainer.__dict__["model"] = model
        metrics = LatentVTONPatchForcingTrainer.garment_gradient_norms(trainer)
        for block_index, scale in enumerate(("dino", "coarse", "middle", "detail"), start=1):
            prefix = f"garment_grad/block_{block_index:02d}_{scale}"
            for projection in ("out_proj", "q", "k", "v"):
                self.assertIn(f"{prefix}/{projection}", metrics)
                self.assertGreater(metrics[f"{prefix}/{projection}"].item(), 0)
        for scale in ("dino", "coarse", "middle", "detail"):
            self.assertGreater(metrics[f"garment_grad/embedder_{scale}"].item(), 0)

    def test_garment_attention_uses_small_nonzero_output_initialization(self):
        model = self._multiscale_model(garment_attention_output_init_std=1e-3)
        for block in model.blocks:
            weight = block.garment_cross_attention.out_proj.weight
            self.assertGreater(weight.abs().sum().item(), 0)
            self.assertLess(weight.std().item(), 2e-3)

    def test_detail_branch_keeps_its_finer_token_grid(self):
        model = self._multiscale_model().eval()
        latent = torch.randn(1, 4, 8, 6)
        position = model._position_embedding(8, 6, latent.dtype, latent.device)
        x = torch.zeros(1, 12, 64)
        tokens, grids = model._garment_branches(
            torch.randn(1, 32, 5, 4),
            torch.randn(1, 4, 8, 6),
            torch.randn(1, 16, 16, 12),
            torch.randn(1, 8, 32, 24),
            x,
            position,
            8,
            6,
        )
        # dino/coarse/middle align with the person token grid; detail is 4x finer in area.
        self.assertEqual(grids["dino"], (4, 3))
        self.assertEqual(grids["coarse"], (4, 3))
        self.assertEqual(grids["middle"], (4, 3))
        self.assertEqual(grids["detail"], (8, 6))
        self.assertEqual(tokens["detail"].shape[1], 48)
        self.assertEqual(tokens["coarse"].shape[1], 12)

    def test_routes_must_reference_enabled_branches(self):
        with self.assertRaises(ValueError):
            self._multiscale_model(garment_scale_routes=["dino", "coarse", "middle", "nope"])
        with self.assertRaises(ValueError):
            self._multiscale_model(
                use_dino_garment=False,
                garment_scale_routes=["dino", "coarse", "middle", "detail"],
            )

    def test_unit_embed_gain_keeps_garment_attention_non_uniform(self):
        """The 0.1 gain flattened the cross-attention logits to a uniform average over
        every garment token, so only the mean garment feature reached the backbone."""
        torch.manual_seed(0)

        # 32x24 garment tokens: the real 512x384 configuration.
        grid_height, grid_width = 32, 24
        keys_count = grid_height * grid_width

        def peak_attention(gain):
            torch.manual_seed(0)
            model = VTONPatchForcingDiT(
                input_size=8,
                patch_size=2,
                in_channels=4,
                hidden_size=64,
                depth=1,
                num_heads=4,
                num_classes=10,
                predict_uncertainty=True,
                garment_feature_dim=32,
                use_vae_garment=False,
                garment_embed_gain=gain,
                cross_attention_every=1,
                compile=False,
            ).eval()
            block = model.blocks[0]
            features = torch.randn(1, 32, grid_height, grid_width).flatten(2).transpose(1, 2)
            keys = model.garment_feature_proj(model.garment_feature_norm(features))
            queries = block.garment_norm(torch.randn(1, 16, 64))
            weight, bias = block.garment_cross_attention.in_proj_weight, block.garment_cross_attention.in_proj_bias
            q = queries @ weight[:64].T + bias[:64]
            k = keys @ weight[64:128].T + bias[64:128]
            logits = (q @ k.transpose(-1, -2)) / (64 / 4) ** 0.5
            return logits.softmax(-1).max().item()

        uniform = 1.0 / keys_count
        attenuated = peak_attention(0.1)
        unit = peak_attention(1.0)
        self.assertLess(attenuated, uniform * 2)
        self.assertGreater(unit, uniform * 5)
        self.assertGreater(unit / attenuated, 3)

    def test_edit_mask_is_not_grown_past_the_token_grid(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        raw_mask = torch.zeros(1, 1, 8, 8)
        raw_mask[:, :, 2:6, 2:6] = 1
        masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32)
        # Token-grid rounding is the only permitted growth; a dilated envelope would
        # force the model to re-synthesise identity evidence it could otherwise copy.
        expected_tokens = torch.nn.functional.adaptive_max_pool2d(raw_mask, (4, 4)) > 0
        torch.testing.assert_close(masks.token, expected_tokens.flatten(2).squeeze(1))
        self.assertEqual(masks.latent.sum().item(), raw_mask.sum().item())

    def test_generate_conditions_on_the_single_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        raw_mask = torch.zeros(1, 1, 8, 8)
        raw_mask[:, :, 2:6, 2:6] = 1
        masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32)
        person_context = torch.ones(1, 4, 8, 8) * (1 - masks.latent)
        model = RecordingVelocityModel()
        flow.generate(
            model,
            torch.randn_like(person_context),
            person_context,
            raw_mask,
            garment=torch.zeros_like(person_context),
            person_condition=person_context,
            person_condition_mask=masks.condition,
            num_steps=1,
        )
        torch.testing.assert_close(model.person_mask, masks.condition)
        torch.testing.assert_close(model.edit_mask, masks.condition)

    def test_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        agnostic = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(agnostic)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            noise,
            agnostic,
            mask,
            garment=torch.zeros_like(agnostic),
            num_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_adaptive_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        agnostic = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            torch.randn_like(agnostic),
            agnostic,
            mask,
            garment=torch.zeros_like(agnostic),
            num_steps=2,
            adaptive=True,
            inner_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_interpolants_use_clean_context_times(self):
        flow = VTONPatchFlowForcing(patch_size=2)
        target = torch.randn(1, 4, 8, 8)
        agnostic = torch.randn_like(target)
        noise = torch.randn_like(target)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        sampled_time = torch.full((1, 16), 0.25)
        xt, _, effective_time, masks = flow.get_interpolants(target, agnostic, mask, x0=noise, t=sampled_time)
        self.assertTrue(torch.all(effective_time.masked_select(~masks.token) == 1))
        torch.testing.assert_close(
            xt.masked_select(~masks.latent.bool()),
            agnostic.masked_select(~masks.latent.bool()),
        )

    def test_zero_time_forcing_uses_pure_noise_in_edit_region(self):
        flow = VTONPatchFlowForcing(patch_size=2, zero_time_probability=1.0)
        target = torch.randn(2, 4, 8, 8)
        context = torch.randn_like(target)
        noise = torch.randn_like(target)
        mask = torch.zeros(2, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        xt, _, timesteps, masks = flow.get_interpolants(target, context, mask, x0=noise)
        self.assertTrue(torch.all(timesteps.masked_select(masks.token) == 0))
        torch.testing.assert_close(xt.masked_select(masks.latent.bool()), noise.masked_select(masks.latent.bool()))

    def test_dino_garment_projection_receives_gradients(self):
        model = VTONPatchForcingDiT(
            input_size=8,
            patch_size=2,
            in_channels=4,
            hidden_size=64,
            depth=1,
            num_heads=4,
            num_classes=10,
            predict_uncertainty=True,
            garment_feature_dim=32,
            cross_attention_every=1,
            compile=False,
        ).train()
        with torch.no_grad():
            model.final_layer.linear.weight.normal_(std=0.01)
            model.blocks[0].garment_cross_attention.out_proj.weight.normal_(std=0.01)
        latent = torch.randn(1, 4, 8, 6)
        mask = torch.ones(1, 1, 8, 6)
        output = model(
            latent,
            torch.rand(1, 12),
            torch.zeros(1, dtype=torch.long),
            person_agnostic=torch.randn_like(latent),
            person_mask=mask,
            edit_mask=mask,
            garment_features=torch.randn(1, 32, 5, 4),
            garment_mask=mask,
        )
        output.square().mean().backward()
        self.assertGreater(model.garment_feature_proj.weight.grad.abs().sum().item(), 0)

    @staticmethod
    def _ltg_sampler(loc, std):
        return {
            "target": "patch_flow.timestep_schedules.LogitNormalTruncatedGaussian",
            "params": {"loc": loc, "std": std, "scale": 1.0},
        }

    def test_high_time_probability_covers_the_detail_regime(self):
        torch.manual_seed(0)
        # The shipped-before configuration: LTG ceiling sigma(0.7+z) plus 30% zero-time.
        starved = VTONPatchFlowForcing(
            patch_size=2,
            timestep_sampler=self._ltg_sampler(0.7, 0.6),
            zero_time_probability=0.3,
        )
        balanced = VTONPatchFlowForcing(
            patch_size=2,
            timestep_sampler=self._ltg_sampler(0.5, 0.25),
            zero_time_probability=0.1,
            high_time_probability=0.25,
            high_time_spread=0.25,
        )
        starved_times = starved._sample_token_times(2048, 64, "cpu", torch.float32)
        balanced_times = balanced._sample_token_times(2048, 64, "cpu", torch.float32)
        starved_detail = (starved_times > 0.9).float().mean().item()
        balanced_detail = (balanced_times > 0.9).float().mean().item()
        # The LTG ceiling alone leaves t>0.9 almost untrained, which is where logo and
        # printed-text structure is written.
        self.assertLess(starved_detail, 0.01)
        self.assertGreater(balanced_detail, 0.05)
        # Garment forcing must survive the rebalance.
        self.assertGreater((balanced_times == 0).float().mean().item(), 0.05)

    def test_high_times_stay_in_range(self):
        torch.manual_seed(0)
        flow = VTONPatchFlowForcing(patch_size=2, high_time_probability=1.0)
        times = flow._sample_token_times(64, 64, "cpu", torch.float32)
        self.assertGreaterEqual(times.min().item(), 0.0)
        self.assertLessEqual(times.max().item(), 1.0)

    def test_rgb_composite_is_exact_outside_mask(self):
        generated = torch.ones(1, 3, 16, 16)
        person = -torch.ones_like(generated)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1
        output = compose_vton(generated, person, mask, feather_radius=2)
        torch.testing.assert_close(output.masked_select(~mask.bool()), person.masked_select(~mask.bool()))


if __name__ == "__main__":
    unittest.main()
