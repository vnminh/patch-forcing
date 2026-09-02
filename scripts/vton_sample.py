import argparse
import os
import sys

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.utils import save_image

current_dir = os.path.dirname(__file__)
repo_dir = os.path.dirname(current_dir)
sys.path.insert(0, repo_dir)

from jutils import instantiate_from_config

from patch_flow.models.pf_transformer_vton import VTONPatchForcingDiT
from patch_flow.vton_data import _letterbox
from patch_flow.vton_utils import compose_vton


def load_rgb(path, size):
    image = Image.open(path).convert("RGB")
    image = _letterbox(image, size, (127, 127, 127), Image.Resampling.BICUBIC)
    return (TF.to_tensor(image) * 2 - 1)[None]


def load_mask(path, size):
    image = Image.open(path).convert("L")
    image = _letterbox(image, size, 0, Image.Resampling.NEAREST)
    return (TF.to_tensor(image) > 0.5).float()[None]


def config_without_pretrained(config):
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if "pretrained_ckpt" in config.params:
        config.params.pretrained_ckpt = None
    return config


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PFT-XL inference")
    device = torch.device("cuda")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    hyper_parameters = checkpoint.get("hyper_parameters")
    if hyper_parameters is None:
        raise ValueError("Expected a VTON training checkpoint containing hyper_parameters")
    model = instantiate_from_config(config_without_pretrained(hyper_parameters["model"]))
    state = VTONPatchForcingDiT._select_checkpoint_state(checkpoint, use_ema=not args.no_ema)
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    autoencoder = instantiate_from_config(hyper_parameters["first_stage"]).to(device).eval()
    flow = instantiate_from_config(hyper_parameters["flow"])

    image_size = (args.height, args.width) if args.height is not None else args.image_size
    person_image = load_rgb(args.person, image_size).to(device)
    garment_image = load_rgb(args.garment, image_size).to(device)
    edit_mask = load_mask(args.agnostic_mask, image_size).to(device)
    garment_mask = load_mask(args.garment_mask, image_size).to(device) if args.garment_mask else None
    person_agnostic_image = person_image * (1 - edit_mask)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        agnostic_latent = autoencoder.encode(person_agnostic_image)
        edit_masks = flow.prepare_masks(edit_mask, agnostic_latent.shape[-2:], agnostic_latent.dtype)
        remove_masks = flow.prepare_masks(
            edit_mask,
            agnostic_latent.shape[-2:],
            agnostic_latent.dtype,
            dilation_tokens=0,
        )
        person_condition = agnostic_latent * (1 - remove_masks.latent)
        person_context = agnostic_latent * (1 - edit_masks.latent)
        garment = autoencoder.encode(garment_image)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        noise = torch.randn(
            person_context.shape,
            generator=generator,
            device=device,
            dtype=person_context.dtype,
        )
        label = torch.full((1,), model.y_embedder.num_classes, device=device, dtype=torch.long)
        sample = flow.generate(
            model=model,
            x=noise,
            person_agnostic=person_context,
            person_condition=person_condition,
            person_condition_mask=remove_masks.condition,
            edit_mask=edit_mask,
            garment=garment,
            garment_mask=garment_mask,
            y=label,
            num_steps=args.steps,
            cfg_scale=args.cfg_scale,
            adaptive=args.adaptive,
            uncertain_fraction=args.uncertain_fraction,
            inner_steps=args.inner_steps,
            progress=True,
        )
        generated = autoencoder.decode(sample)
        expanded = edit_masks.latent
        expanded = torch.nn.functional.interpolate(expanded, size=person_image.shape[-2:], mode="nearest")
        output = compose_vton(generated, person_image, expanded, feather_radius=args.feather_radius)
    save_image(output.float(), args.output, normalize=True, value_range=(-1, 1))
    print(f"Saved try-on result to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--person", required=True)
    parser.add_argument("--garment", required=True)
    parser.add_argument("--agnostic-mask", required=True)
    parser.add_argument("--garment-mask")
    parser.add_argument("--output", default="vton_result.png")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--uncertain-fraction", type=float, default=0.3)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--feather-radius", type=int, default=8)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    if (args.height is None) != (args.width is None):
        parser.error("--height and --width must be specified together")
    main(args)
