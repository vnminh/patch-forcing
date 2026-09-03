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
from patch_flow.vton_utils import compose_vton, prepare_vton_masks
from patch_flow.vton_data import VTONHDDataset
from train import repeat_dataloader


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
    def test_only_requested_dino_tail_is_trainable(self):
        class FakeDino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(12)])
                self.embeddings = torch.nn.Linear(4, 4)
                self.layernorm = torch.nn.LayerNorm(4)
                self.config = SimpleNamespace(patch_size=14, hidden_size=4)

        with patch("patch_flow.dino_garment.Dinov2Model.from_pretrained", return_value=FakeDino()):
            encoder = TrainableDinoGarmentEncoder(trainable_blocks=2)
        frozen = encoder.model.encoder.layer[:10]
        trainable = encoder.model.encoder.layer[10:]
        self.assertTrue(all(not parameter.requires_grad for layer in frozen for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for layer in trainable for parameter in layer.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in encoder.model.layernorm.parameters()))

    def test_full_dino_finetuning_includes_embeddings(self):
        class FakeDino(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(12)])
                self.embeddings = torch.nn.Linear(4, 4)
                self.layernorm = torch.nn.LayerNorm(4)
                self.config = SimpleNamespace(patch_size=14, hidden_size=4)

        with patch("patch_flow.dino_garment.Dinov2Model.from_pretrained", return_value=FakeDino()):
            encoder = TrainableDinoGarmentEncoder(trainable_blocks=12)
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

    def test_expanded_edit_mask_does_not_erase_person_condition(self):
        flow = VTONPatchFlowForcing(patch_size=2, mask_dilation_tokens=1)
        raw_mask = torch.zeros(1, 1, 8, 8)
        raw_mask[:, :, 2:6, 2:6] = 1
        remove_masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32, dilation_tokens=0)
        edit_masks = flow.prepare_masks(raw_mask, (8, 8), torch.float32, dilation_tokens=1)
        person_condition = torch.ones(1, 4, 8, 8) * (1 - remove_masks.latent)
        person_context = person_condition * (1 - edit_masks.latent)
        model = RecordingVelocityModel()
        flow.generate(
            model,
            torch.randn_like(person_context),
            person_context,
            raw_mask,
            torch.zeros_like(person_context),
            person_condition=person_condition,
            person_condition_mask=remove_masks.condition,
            num_steps=1,
        )
        ring = edit_masks.latent.bool() & ~remove_masks.latent.bool()
        self.assertGreater(model.person_condition.masked_select(ring).abs().sum().item(), 0)
        torch.testing.assert_close(model.person_mask, remove_masks.condition)
        torch.testing.assert_close(model.edit_mask, edit_masks.condition)

    def test_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2, mask_dilation_tokens=0)
        agnostic = torch.randn(1, 4, 8, 8)
        noise = torch.randn_like(agnostic)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            noise,
            agnostic,
            mask,
            torch.zeros_like(agnostic),
            num_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2, dilation_tokens=0).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_adaptive_sampler_preserves_latent_outside_mask(self):
        flow = VTONPatchFlowForcing(patch_size=2, mask_dilation_tokens=0)
        agnostic = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:6, 2:6] = 1
        output = flow.generate(
            ConstantVelocityModel(),
            torch.randn_like(agnostic),
            agnostic,
            mask,
            torch.zeros_like(agnostic),
            num_steps=2,
            adaptive=True,
            inner_steps=2,
        )
        latent_mask = prepare_vton_masks(mask, (8, 8), patch_size=2, dilation_tokens=0).latent.bool()
        torch.testing.assert_close(output.masked_select(~latent_mask), agnostic.masked_select(~latent_mask))

    def test_interpolants_use_clean_context_times(self):
        flow = VTONPatchFlowForcing(patch_size=2, mask_dilation_tokens=0)
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
        flow = VTONPatchFlowForcing(
            patch_size=2,
            mask_dilation_tokens=0,
            zero_time_probability=1.0,
        )
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

    def test_one_token_dilation(self):
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:8, 4:8] = 1
        no_dilation = prepare_vton_masks(mask, (8, 8), patch_size=2, dilation_tokens=0)
        dilated = prepare_vton_masks(mask, (8, 8), patch_size=2, dilation_tokens=1)
        self.assertGreater(dilated.token.sum(), no_dilation.token.sum())

    def test_rgb_composite_is_exact_outside_mask(self):
        generated = torch.ones(1, 3, 16, 16)
        person = -torch.ones_like(generated)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, 4:12, 4:12] = 1
        output = compose_vton(generated, person, mask, feather_radius=2)
        torch.testing.assert_close(output.masked_select(~mask.bool()), person.masked_select(~mask.bool()))


if __name__ == "__main__":
    unittest.main()
