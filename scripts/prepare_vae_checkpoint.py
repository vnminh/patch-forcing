import argparse
import os

import torch

from jutils.nn.kl_autoencoder import AutoencoderKL


PREFIXES = (
    "module.",
    "model.first_stage_model.",
    "first_stage_model.",
    "model.",
)


def key_variants(key):
    variants = {key}
    changed = True
    while changed:
        changed = False
        for variant in tuple(variants):
            for prefix in PREFIXES:
                if variant.startswith(prefix):
                    stripped = variant[len(prefix) :]
                    if stripped not in variants:
                        variants.add(stripped)
                        changed = True
    return variants


def main(args):
    checkpoint = torch.load(args.input, map_location="cpu", weights_only=True)
    source = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source, dict):
        raise TypeError("VAE checkpoint does not contain a state dictionary")

    model = AutoencoderKL(ckpt_path=None)
    expected = model.state_dict()
    converted = {}
    for source_key, value in source.items():
        for candidate in key_variants(source_key):
            if candidate in expected and expected[candidate].shape == value.shape:
                converted[candidate] = value
                break

    missing = sorted(set(expected) - set(converted))
    if missing:
        raise RuntimeError(f"VAE conversion is missing {len(missing)} keys; first keys: {missing[:10]}")
    model.load_state_dict(converted, strict=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(converted, args.output)
    print(f"Saved {len(converted)} VAE tensors to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())

