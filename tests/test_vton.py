import unittest

import torch

from patch_flow.flow_vton import VTONPatchFlowForcing
from patch_flow.models.pf_transformer import PatchForcingDiT
from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.vton_utils import compose_vton, prepare_vton_masks
from train import repeat_dataloader


class ConstantVelocityModel(torch.nn.Module):
    def forward(self, x, t, return_uncertainty=False, **kwargs):
        velocity = torch.ones_like(x)
        if return_uncertainty:
            return velocity, torch.zeros_like(x[:, :1])
        return velocity


class VTONTests(unittest.TestCase):
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
                garment=zeros,
                garment_mask=mask,
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
            garment=torch.randn_like(x),
            garment_mask=mask,
        )
        output.square().mean().backward()
        self.assertGreater(model.x_embedder.proj.weight.grad[:, 4:].abs().sum().item(), 0)
        cross = model.blocks[0].garment_cross_attention.out_proj.weight.grad
        self.assertGreater(cross.abs().sum().item(), 0)

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
