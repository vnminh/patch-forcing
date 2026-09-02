#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
deployment_root=$(cd "$repo_root/.." && pwd)
dataset_root=${VITONHD_ROOT:-$deployment_root/high-resolution-viton-zalando-dataset}
smoke_dir=${VITONHD_SMOKE_DIR:-$dataset_root/smoke32}
checkpoints_dir=${CHECKPOINTS_DIR:-$repo_root/checkpoints}
venv_path=${PFT_VENV:-/venv/main}
service_name=${PFT_SERVICE_NAME:-pft-vton-training}
experiment=${PFT_EXPERIMENT:-viton-pft-xl-smoke16gb}
train_overrides=${PFT_TRAIN_OVERRIDES:-}
environment_file="$deployment_root/training.env"
supervisor_config="/etc/supervisor/conf.d/$service_name.conf"

if [[ $EUID -ne 0 ]]; then
    echo "Run this script as root so it can register the Supervisor job." >&2
    exit 1
fi
for path in "$repo_root" "$deployment_root" "$dataset_root" "$venv_path"; do
    if [[ "$path" =~ [[:space:]] ]]; then
        echo "Paths containing whitespace are not supported: $path" >&2
        exit 2
    fi
done
if [[ ! -f "$venv_path/bin/activate" ]]; then
    echo "Python environment not found: $venv_path" >&2
    exit 1
fi
for split in train test; do
    for directory in image cloth agnostic-mask cloth-mask; do
        if [[ ! -d "$dataset_root/$split/$directory" ]]; then
            echo "Missing remote dataset directory: $dataset_root/$split/$directory" >&2
            exit 1
        fi
    done
done

# shellcheck disable=SC1091
source "$venv_path/bin/activate"
export VITONHD_ROOT="$dataset_root"
export VITONHD_SMOKE_DIR="$smoke_dir"
export PFT_XL_CKPT="$checkpoints_dir/pft-xl_step400k_ema.ckpt"

dependency_check=0
python - <<'PY' || dependency_check=$?
import accelerate
import cv2
import diffusers
import einops
import hydra
import lightning
import timm
import torch
import torchvision
import transformers
import jutils
assert torch.cuda.is_available(), "CUDA is not available"
print(f"Using torch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}")
PY

setup_args=(USE_CURRENT_ENV=1 CHECKPOINTS_DIR="$checkpoints_dir")
if [[ $dependency_check -eq 0 ]]; then
    setup_args+=(SKIP_INSTALL=1)
else
    echo "Installing the pinned training environment into $venv_path"
fi
env "${setup_args[@]}" bash "$repo_root/setup.sh"

python "$repo_root/scripts/make_vton_smoke_split.py" \
    --dataset-root "$dataset_root" \
    --output-dir "$smoke_dir"

mkdir -p "$deployment_root/logs"
{
    printf 'export VITONHD_ROOT=%q\n' "$dataset_root"
    printf 'export VITONHD_SMOKE_DIR=%q\n' "$smoke_dir"
    printf 'export PFT_XL_CKPT=%q\n' "$checkpoints_dir/pft-xl_step400k_ema.ckpt"
    printf 'export PFT_VENV=%q\n' "$venv_path"
    printf 'export PFT_EXPERIMENT=%q\n' "$experiment"
    printf 'export PFT_TRAIN_OVERRIDES=%q\n' "$train_overrides"
} > "$environment_file"
chmod 600 "$environment_file"

cat > "$supervisor_config" <<EOF
[program:$service_name]
command=/usr/bin/env bash $repo_root/scripts/vton_training_process.sh
directory=$repo_root
autostart=false
autorestart=unexpected
startsecs=10
startretries=2
stopasgroup=true
killasgroup=true
stopsignal=INT
stopwaitsecs=30
stdout_logfile=$deployment_root/logs/$service_name.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=3
redirect_stderr=true
environment=PROC_NAME="%(program_name)s"
EOF

chmod +x "$repo_root/scripts/vton_training_process.sh"
supervisorctl reread
supervisorctl update
if supervisorctl status "$service_name" 2>/dev/null | grep -Eq 'RUNNING|STARTING'; then
    supervisorctl restart "$service_name"
else
    supervisorctl start "$service_name"
fi

echo "Supervisor service: $service_name"
echo "Training log: $deployment_root/logs/$service_name.log"
echo "TensorBoard logs: $repo_root/logs"
echo "Status: supervisorctl status $service_name"

