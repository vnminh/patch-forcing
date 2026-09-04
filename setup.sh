#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_root"

env_name=${ENV_NAME:-pft-vton}
checkpoints_dir=${CHECKPOINTS_DIR:-$repo_root/checkpoints}
pft_path="$checkpoints_dir/pft-xl_step400k_ema.ckpt"
vae_original_path="$checkpoints_dir/vae-ft-ema-560000-ema-pruned.ckpt"
vae_path="$checkpoints_dir/sd_ae.ckpt"
pft_url=${PFT_XL_URL:-https://ommer-lab.com/files/pft/pft-xl_step400k_ema.ckpt}
vae_url=${SD_VAE_URL:-https://huggingface.co/stabilityai/sd-vae-ft-ema-original/resolve/main/vae-ft-ema-560000-ema-pruned.ckpt}
teacher_model=${CORRESPONDENCE_TEACHER:-facebook/dinov3-vits16-pretrain-lvd1689m}
vae_sha256=0b204ad0cae549e0a7e298d803d57e36363760dec71c63109c1da3e1147ec520
activation_command="current environment"

activate_environment() {
    if [[ ${USE_CURRENT_ENV:-0} == 1 ]]; then
        return
    fi
    if command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
        if ! conda env list | awk '{print $1}' | grep -Fxq "$env_name"; then
            conda create -y -n "$env_name" python=3.12
        fi
        conda activate "$env_name"
        activation_command="conda activate $env_name"
        return
    fi
    local venv_path="$repo_root/.venv"
    if [[ ! -x "$venv_path/bin/python" ]]; then
        python3 -m venv "$venv_path"
    fi
    source "$venv_path/bin/activate"
    activation_command="source $venv_path/bin/activate"
}

download_file() {
    local url=$1
    local destination=$2
    if [[ -s "$destination" && ${FORCE_DOWNLOAD:-0} != 1 ]]; then
        echo "Using existing $destination"
        return
    fi
    if [[ ${FORCE_DOWNLOAD:-0} == 1 ]]; then
        rm -f "$destination" "$destination.part"
    fi
    curl \
        --fail \
        --location \
        --retry 5 \
        --retry-delay 5 \
        --continue-at - \
        --output "$destination.part" \
        "$url"
    mv "$destination.part" "$destination"
}

activate_environment

python -c 'import sys; assert sys.version_info[:2] == (3, 12), f"Python 3.12 is required, got {sys.version}"'

if [[ ${SKIP_INSTALL:-0} != 1 ]]; then
    python -m pip install --upgrade pip setuptools wheel
    python -m pip install \
        torch==2.8.0+cu128 \
        torchvision==0.23.0+cu128 \
        --index-url https://download.pytorch.org/whl/cu128
    python -m pip install -r requirements.txt
    if [[ ${INSTALL_FLASH_ATTN:-0} == 1 ]]; then
        python -m pip install flash-attn==2.8.3 --no-build-isolation
    fi
    python -m pip check
fi

mkdir -p "$checkpoints_dir"
download_file "$pft_url" "$pft_path"
download_file "$vae_url" "$vae_original_path"
# The correspondence teacher is training-time only, and its repository is gated, so a
# failure here is a warning rather than a setup failure: inference never needs it, and
# training can run with correspondence_center_weight=0.
CORRESPONDENCE_TEACHER="$teacher_model" python -c 'import os; from transformers import AutoModel; AutoModel.from_pretrained(os.environ["CORRESPONDENCE_TEACHER"])' || {
    echo "Warning: could not fetch $teacher_model."
    echo "It is gated on Hugging Face; set HF_TOKEN for an account with access,"
    echo "or train with trainer.params.correspondence_center_weight=0 and"
    echo "trainer.params.correspondence_entropy_weight=0."
}

echo "$vae_sha256  $vae_original_path" | sha256sum --check --status || {
    echo "VAE checksum verification failed: $vae_original_path"
    echo "Run with FORCE_DOWNLOAD=1 to download it again."
    exit 1
}

if [[ ! -s "$vae_path" || ${FORCE_CONVERT:-0} == 1 ]]; then
    python scripts/prepare_vae_checkpoint.py \
        --input "$vae_original_path" \
        --output "$vae_path"
fi

if [[ ${SKIP_VERIFY:-0} != 1 ]]; then
    python scripts/verify_vton_setup.py --pft "$pft_path" --vae "$vae_path"
fi

echo
echo "Setup complete."
echo "Activate with: $activation_command"
echo "Set PFT_XL_CKPT=$pft_path"
echo "VAE checkpoint: $vae_path"
echo "Correspondence teacher: $teacher_model"
