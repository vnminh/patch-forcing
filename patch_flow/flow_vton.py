import torch
import torch.nn.functional as F

from jutils import instantiate_from_config

from patch_flow.vton_utils import prepare_vton_masks


class VTONPatchFlowForcing:
    def __init__(
        self,
        timestep_sampler=None,
        patch_size=2,
        mask_dilation_tokens=1,
        mask_dilation_jitter_tokens=0,
    ):
        self.patch_size = int(patch_size)
        self.mask_dilation_tokens = int(mask_dilation_tokens)
        self.mask_dilation_jitter_tokens = int(mask_dilation_jitter_tokens)
        self.t_sampler = torch.rand if timestep_sampler is None else instantiate_from_config(timestep_sampler)

    def sample_training_dilation(self):
        dilation = self.mask_dilation_tokens
        if self.mask_dilation_jitter_tokens > 0:
            dilation += int(torch.randint(self.mask_dilation_jitter_tokens + 1, ()).item())
        return dilation

    def prepare_masks(self, edit_mask, latent_size, dtype, dilation_tokens=None):
        masks = prepare_vton_masks(
            edit_mask,
            latent_size,
            patch_size=self.patch_size,
            dilation_tokens=self.mask_dilation_tokens if dilation_tokens is None else dilation_tokens,
        )
        masks.condition = masks.condition.to(dtype)
        masks.latent = masks.latent.to(dtype)
        return masks

    def get_interpolants(self, x1, person_context, edit_mask, x0=None, t=None, masks=None):
        if x1.shape != person_context.shape:
            raise ValueError(
                f"Target and context latents must match, got {x1.shape} and {person_context.shape}"
            )
        if x0 is None:
            x0 = torch.randn_like(x1)
        if masks is None:
            masks = self.prepare_masks(
                edit_mask,
                x1.shape[-2:],
                x1.dtype,
                dilation_tokens=self.sample_training_dilation(),
            )
        batch, _, height, width = x1.shape
        tokens = (height // self.patch_size) * (width // self.patch_size)
        if t is None:
            t = self.t_sampler((batch, tokens), device=x1.device, dtype=x1.dtype)
        if t.shape != (batch, tokens):
            raise ValueError(f"Expected timestep shape {(batch, tokens)}, got {tuple(t.shape)}")
        t_effective = torch.where(masks.token, t, torch.ones_like(t))
        t_latent = t_effective.view(batch, 1, height // self.patch_size, width // self.patch_size)
        t_latent = t_latent.repeat_interleave(self.patch_size, -2).repeat_interleave(self.patch_size, -1)
        interpolated = t_latent * x1 + (1 - t_latent) * x0
        xt = masks.latent * interpolated + (1 - masks.latent) * person_context
        ut = x1 - x0
        return xt, ut, t_effective, masks

    @staticmethod
    def _repeat_condition(value, repeats):
        if value is None:
            return None
        return torch.cat([value] * repeats, dim=0)

    def _predict(
        self,
        model,
        x,
        t,
        person_condition,
        person_condition_mask,
        edit_condition_mask,
        garment,
        garment_mask,
        y,
        cfg_scale,
        return_uncertainty,
    ):
        kwargs = dict(
            person_agnostic=person_condition,
            person_mask=person_condition_mask,
            edit_mask=edit_condition_mask,
            garment=garment,
            garment_mask=garment_mask,
            y=y,
            return_uncertainty=return_uncertainty,
        )
        if cfg_scale == 1.0 or garment is None:
            return model(x=x, t=t, **kwargs)

        batch = x.shape[0]
        x_in = torch.cat((x, x), dim=0)
        t_in = torch.cat((t, t), dim=0)
        kwargs["person_agnostic"] = self._repeat_condition(person_condition, 2)
        kwargs["person_mask"] = self._repeat_condition(person_condition_mask, 2)
        kwargs["edit_mask"] = self._repeat_condition(edit_condition_mask, 2)
        kwargs["y"] = self._repeat_condition(y, 2)
        kwargs["garment"] = torch.cat((torch.zeros_like(garment), garment), dim=0)
        if garment_mask is not None:
            kwargs["garment_mask"] = torch.cat((torch.zeros_like(garment_mask), garment_mask), dim=0)
        output = model(x=x_in, t=t_in, **kwargs)
        if return_uncertainty:
            velocity, uncertainty = output
            velocity_uncond, velocity_cond = velocity.chunk(2)
            _, uncertainty_cond = uncertainty.chunk(2)
            velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
            return velocity, uncertainty_cond
        velocity_uncond, velocity_cond = output.chunk(2)
        if velocity_uncond.shape[0] != batch:
            raise RuntimeError("Invalid classifier-free guidance batch")
        return velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)

    def _uncertain_tokens(self, logvar, edit_tokens, fraction):
        pooled = F.avg_pool2d(logvar.float(), kernel_size=self.patch_size, stride=self.patch_size).flatten(1)
        uncertain = torch.zeros_like(edit_tokens)
        for batch_index in range(edit_tokens.shape[0]):
            indices = torch.where(edit_tokens[batch_index])[0]
            if indices.numel() == 0:
                continue
            count = max(1, round(indices.numel() * fraction))
            selected = indices[torch.topk(pooled[batch_index, indices], k=count).indices]
            uncertain[batch_index, selected] = True
        return uncertain

    def _tokens_to_latent(self, tokens, height, width, dtype):
        grid = tokens.view(tokens.shape[0], 1, height // self.patch_size, width // self.patch_size)
        return grid.repeat_interleave(self.patch_size, -2).repeat_interleave(self.patch_size, -1).to(dtype)

    @torch.no_grad()
    def generate(
        self,
        model,
        x,
        person_agnostic,
        edit_mask,
        garment,
        garment_mask=None,
        person_condition=None,
        person_condition_mask=None,
        y=None,
        num_steps=50,
        cfg_scale=1.0,
        adaptive=False,
        uncertain_fraction=0.3,
        inner_steps=3,
        progress=False,
    ):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not 0 < uncertain_fraction <= 1:
            raise ValueError("uncertain_fraction must be in (0, 1]")
        if adaptive and inner_steps < 2:
            raise ValueError("Adaptive sampling requires at least two inner steps")
        masks = self.prepare_masks(edit_mask, person_agnostic.shape[-2:], person_agnostic.dtype)
        batch, _, height, width = person_agnostic.shape
        person_context = person_agnostic * (1 - masks.latent)
        if person_condition is None:
            person_condition = person_context
        if person_condition_mask is None:
            person_condition_mask = masks.condition
        xt = masks.latent * x + (1 - masks.latent) * person_context
        token_times = torch.where(
            masks.token,
            torch.zeros_like(masks.token, dtype=x.dtype),
            torch.ones_like(masks.token, dtype=x.dtype),
        )
        time_grid = torch.linspace(0, 1, num_steps + 1, device=x.device, dtype=x.dtype)
        iterator = zip(time_grid[:-1], time_grid[1:])
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=num_steps)

        for current, next_time in iterator:
            output = self._predict(
                model,
                xt,
                token_times,
                person_condition,
                person_condition_mask,
                masks.condition,
                garment,
                garment_mask,
                y,
                cfg_scale,
                return_uncertainty=adaptive,
            )
            if adaptive:
                velocity, logvar = output
                uncertain_tokens = self._uncertain_tokens(logvar, masks.token, uncertain_fraction)
            else:
                velocity = output
                uncertain_tokens = torch.zeros_like(masks.token)

            easy_tokens = masks.token & ~uncertain_tokens
            easy_latent = self._tokens_to_latent(easy_tokens, height, width, x.dtype)
            uncertain_latent = self._tokens_to_latent(uncertain_tokens, height, width, x.dtype)
            delta = next_time - current
            xt = xt + delta * velocity * easy_latent
            if adaptive:
                inner_delta = delta / inner_steps
                xt = xt + inner_delta * velocity * uncertain_latent
                token_times = token_times + delta * easy_tokens.to(x.dtype) + inner_delta * uncertain_tokens.to(x.dtype)
                for _ in range(inner_steps - 1):
                    velocity = self._predict(
                        model,
                        xt,
                        token_times,
                        person_condition,
                        person_condition_mask,
                        masks.condition,
                        garment,
                        garment_mask,
                        y,
                        cfg_scale,
                        return_uncertainty=False,
                    )
                    xt = xt + inner_delta * velocity * uncertain_latent
                    token_times = token_times + inner_delta * uncertain_tokens.to(x.dtype)
            else:
                token_times = token_times + delta * masks.token.to(x.dtype)
                xt = xt + delta * velocity * uncertain_latent
            xt = masks.latent * xt + (1 - masks.latent) * person_context
        return xt
