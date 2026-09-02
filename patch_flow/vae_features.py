import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_vae_pyramid(autoencoder, image):
    encoder = getattr(autoencoder, "encoder", None)
    quant_conv = getattr(autoencoder, "quant_conv", None)
    if encoder is None or quant_conv is None or not hasattr(encoder, "down"):
        raise TypeError("Multiscale garment conditioning requires the jutils SD AutoencoderKL")
    if len(encoder.down) < 3:
        raise ValueError("The VAE encoder must have at least three resolution levels")

    hidden = encoder.conv_in(image)
    detail = None
    middle = None
    for level, down in enumerate(encoder.down):
        for block_index, block in enumerate(down.block):
            hidden = block(hidden, None)
            if len(down.attn) > 0:
                hidden = down.attn[block_index](hidden)
        if hasattr(down, "downsample"):
            hidden = down.downsample(hidden)
            if level == 0:
                detail = hidden
            elif level == 1:
                middle = hidden

    hidden = encoder.mid.block_1(hidden, None)
    hidden = encoder.mid.attn_1(hidden)
    hidden = encoder.mid.block_2(hidden, None)
    hidden = encoder.norm_out(hidden)
    hidden = F.silu(hidden, inplace=True)
    moments = quant_conv(encoder.conv_out(hidden))
    latent = moments.chunk(2, dim=1)[0]
    latent = (latent + float(getattr(autoencoder, "shift", 0.0))) * float(
        getattr(autoencoder, "scale", 1.0)
    )
    if detail is None or middle is None:
        raise RuntimeError("Failed to collect the VAE 1/2 and 1/4 feature maps")
    return latent, middle, detail
