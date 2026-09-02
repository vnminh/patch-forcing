#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
remote_host=${REMOTE_HOST:-root@199.126.134.31}
remote_port=${REMOTE_PORT:-29644}
remote_root=${REMOTE_ROOT:-/workspace/pft-vton}
remote_repo="$remote_root/patch-forcing"

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: scripts/transfer_code_remote.sh

Resumably syncs only the patch-forcing working tree. Optional variables:
REMOTE_HOST, REMOTE_PORT, REMOTE_ROOT, and DRY_RUN=1.
EOF
    exit 0
fi

for command_name in ssh rsync; do
    command -v "$command_name" >/dev/null || {
        echo "Required command is unavailable: $command_name" >&2
        exit 1
    }
done
if [[ ! -f "$repo_root/train.py" ]]; then
    echo "Could not find patch-forcing/train.py under $repo_root" >&2
    exit 1
fi
if [[ ! "$remote_port" =~ ^[0-9]+$ ]]; then
    echo "REMOTE_PORT must be numeric: $remote_port" >&2
    exit 2
fi
if [[ "$remote_host" =~ [[:space:]] || ! "$remote_root" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "REMOTE_HOST or REMOTE_ROOT contains unsupported characters" >&2
    exit 2
fi

ssh_args=(-p "$remote_port" -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
rsync_ssh="ssh -p $remote_port -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6"
rsync_args=(-a --human-readable --info=progress2 --partial --append-verify)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    rsync_args+=(--dry-run)
    echo "Dry run: remote directories and files will not be created."
else
    ssh "${ssh_args[@]}" "$remote_host" mkdir -p "$remote_repo"
fi

echo "Syncing code to $remote_host:$remote_repo"
rsync "${rsync_args[@]}" \
    -e "$rsync_ssh" \
    --exclude .git/ \
    --exclude .venv/ \
    --exclude checkpoints/ \
    --exclude logs/ \
    --exclude outputs/ \
    --exclude results/ \
    --exclude wandb/ \
    --exclude __pycache__/ \
    --exclude .pytest_cache/ \
    "$repo_root/" "$remote_host:$remote_repo/"

echo "Code transfer complete."

