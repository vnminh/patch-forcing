import argparse
import gc

import torch

from jutils.nn.kl_autoencoder import AutoencoderKL


def main(args):
    checkpoint = torch.load(args.pft, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "config" not in checkpoint:
        raise RuntimeError("PFT-XL checkpoint must contain 'state_dict' and 'config'")
    state = checkpoint["state_dict"]
    input_weight = state.get("x_embedder.proj.weight")
    if input_weight is None or input_weight.ndim != 4 or input_weight.shape[1] != 4:
        raise RuntimeError("PFT-XL checkpoint has an incompatible input projection")
    print(f"PFT-XL checkpoint: {len(state)} tensors, input projection {tuple(input_weight.shape)}")
    del checkpoint, state, input_weight
    gc.collect()

    autoencoder = AutoencoderKL(ckpt_path=args.vae)
    print(f"SD VAE checkpoint: {len(autoencoder.state_dict())} tensors")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pft", required=True)
    parser.add_argument("--vae", required=True)
    main(parser.parse_args())

