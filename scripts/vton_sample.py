"""Generate a VITON result from person-side and garment-side conditions."""

import argparse
import os
import random
import sys
from functools import partial

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

pdir = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, pdir)

from jutils import instantiate_from_config
from patch_flow.integrators import EulerPF


def _rgb(path: str, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = TF.resize(image, [height, width], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.to_tensor(image).mul_(2.0).sub_(1.0)


def _mask(path: str, height: int, width: int) -> torch.Tensor:
    image = Image.open(path).convert("L")
    image = TF.resize(image, [height, width], interpolation=InterpolationMode.NEAREST)
    return (TF.to_tensor(image) > 0.5).float()


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    prefix = f"{prefix}."
    return {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}


def _load_module(config_path: str, state_dict: dict, prefix: str, device: torch.device):
    config = OmegaConf.load(config_path)
    module = instantiate_from_config(config).to(device)
    module.load_state_dict(_strip_prefix(state_dict, prefix), strict=True)
    return module.eval()


def main(args):
    if args.height % 16 or args.width % 16:
        raise ValueError("height and width must be divisible by 16 for VAE stride 8 and Patch Forcing patch size 2.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VTON sampling.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    model = _load_module(args.model_config, state_dict, "model", device)
    condition_encoder = _load_module(args.condition_encoder_config, state_dict, "condition_encoder", device)
    vae = instantiate_from_config(OmegaConf.load(args.autoencoder_config)).to(device).eval()

    batch = {
        "agnostic": _rgb(args.agnostic, args.height, args.width).unsqueeze(0).to(device),
        "densepose": _rgb(args.densepose, args.height, args.width).unsqueeze(0).to(device),
        "cloth": _rgb(args.cloth, args.height, args.width).unsqueeze(0).to(device),
        "cloth_mask": _mask(args.cloth_mask, args.height, args.width).unsqueeze(0).to(device),
    }
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        agnostic_latent = vae.encode(batch["agnostic"])
        conditions = condition_encoder(
            agnostic_latent=agnostic_latent,
            densepose=batch["densepose"],
            cloth=batch["cloth"],
            cloth_mask=batch["cloth_mask"],
        )
        latent_shape = (1, model.latent_dim, args.height // 8, args.width // 8)
        noise = torch.randn(latent_shape, device=device)
        timesteps = torch.linspace(0, 1, args.num_steps + 1, device=device)
        sampler = EulerPF(patch_size=model.patch_size)

        kwargs = dict(conditions)
        if args.cfg_scale != 1.0:
            kwargs.update(uc_cond=condition_encoder.unconditional(conditions), cond_key="garment_tokens")
        latent = sampler(
            model=model,
            x=noise,
            timesteps=timesteps,
            progress=True,
            cfg_scale=args.cfg_scale,
            **kwargs,
        )
        sample = vae.decode(latent)

    image = einops.rearrange(sample[0], "c h w -> h w c")
    image = torch.clamp(image * 127.5 + 127.5, 0, 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(image).save(args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--agnostic", required=True)
    parser.add_argument("--densepose", required=True)
    parser.add_argument("--cloth", required=True)
    parser.add_argument("--cloth-mask", required=True)
    parser.add_argument("--output", default="vton_result.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--model-config", default="configs/model/vton-pft-xl.yaml")
    parser.add_argument("--condition-encoder-config", default="configs/condition_encoder/vton-xl.yaml")
    parser.add_argument("--autoencoder-config", default="configs/autoencoder/flux2_ae.yaml")
    args = parser.parse_args()
    main(args)
