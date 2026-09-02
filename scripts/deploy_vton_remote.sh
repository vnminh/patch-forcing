#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
remote_host=${REMOTE_HOST:-root@199.126.134.31}
remote_port=${REMOTE_PORT:-29644}
remote_root=${REMOTE_ROOT:-/workspace/pft-vton}

"$script_dir/transfer_code_remote.sh"
"$script_dir/transfer_data_remote.sh"

if [[ ${DRY_RUN:-0} == 1 ]]; then
    echo "Dry run complete; training was not started."
    exit 0
fi

echo "Preparing and starting remote VTON training"
ssh \
    -p "$remote_port" \
    -o BatchMode=yes \
    -o ServerAliveInterval=15 \
    -o ServerAliveCountMax=6 \
    "$remote_host" \
    "PFT_EXPERIMENT='${PFT_EXPERIMENT:-viton-pft-xl-smoke16gb}' PFT_TRAIN_OVERRIDES='${PFT_TRAIN_OVERRIDES:-}' bash '$remote_root/patch-forcing/scripts/start_vton_training_remote.sh'"

ssh -p "$remote_port" -o BatchMode=yes "$remote_host" \
    "supervisorctl status pft-vton-training; tail -n 40 '$remote_root/logs/pft-vton-training.log'"
