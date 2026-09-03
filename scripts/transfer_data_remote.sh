#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)
local_dataset=${LOCAL_DATASET:-$(cd "$repo_root/.." && pwd)/high-resolution-viton-zalando-dataset}
remote_host=${REMOTE_HOST:-root@199.126.134.31}
remote_port=${REMOTE_PORT:-29644}
remote_root=${REMOTE_ROOT:-/workspace}
remote_dataset="$remote_root/high-resolution-viton-zalando-dataset"
transfer_jobs=${TRANSFER_JOBS:-6}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
    cat <<'EOF'
Usage: scripts/transfer_data_remote.sh

Resumably syncs only the full VITON-HD dataset. Optional variables:
LOCAL_DATASET, REMOTE_HOST, REMOTE_PORT, REMOTE_ROOT, TRANSFER_JOBS (default: 6),
and DRY_RUN=1.
EOF
    exit 0
fi

for command_name in ssh rsync find; do
    command -v "$command_name" >/dev/null || {
        echo "Required command is unavailable: $command_name" >&2
        exit 1
    }
done
if [[ ! -d "$local_dataset" ]]; then
    echo "Dataset directory does not exist: $local_dataset" >&2
    exit 1
fi
for split in train test; do
    for directory in image cloth agnostic-mask cloth-mask; do
        if [[ ! -d "$local_dataset/$split/$directory" ]]; then
            echo "Missing dataset directory: $local_dataset/$split/$directory" >&2
            exit 1
        fi
    done
done
for pair_file in train_pairs.txt test_pairs.txt; do
    if [[ ! -s "$local_dataset/$pair_file" ]]; then
        echo "Missing or empty pair list: $local_dataset/$pair_file" >&2
        exit 1
    fi
done
if [[ ! "$remote_port" =~ ^[0-9]+$ ]]; then
    echo "REMOTE_PORT must be numeric: $remote_port" >&2
    exit 2
fi
if [[ ! "$transfer_jobs" =~ ^[1-9][0-9]*$ || "$transfer_jobs" -gt 32 ]]; then
    echo "TRANSFER_JOBS must be between 1 and 32: $transfer_jobs" >&2
    exit 2
fi
if [[ "$remote_host" =~ [[:space:]] || ! "$remote_root" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
    echo "REMOTE_HOST or REMOTE_ROOT contains unsupported characters" >&2
    exit 2
fi

ssh_args=(-p "$remote_port" -o ClearAllForwardings=yes -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6)
rsync_ssh="ssh -p $remote_port -o ClearAllForwardings=yes -o BatchMode=yes -o ServerAliveInterval=15 -o ServerAliveCountMax=6"
rsync_args=(-a --human-readable --info=progress2 --partial --append-verify)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    rsync_args+=(--dry-run)
    echo "Dry run: remote directories and files will not be created."
else
    ssh "${ssh_args[@]}" "$remote_host" mkdir -p "$remote_dataset"
fi

manifest_dir=$(mktemp -d "${TMPDIR:-/tmp}/vton-transfer.XXXXXX")
worker_pids=()

cleanup_manifests() {
    find "$manifest_dir" -type f -delete 2>/dev/null || true
    rmdir "$manifest_dir" 2>/dev/null || true
}

stop_workers() {
    local pid
    for pid in "${worker_pids[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    for pid in "${worker_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}

interrupted() {
    echo "Stopping all transfer workers; completed files remain resumable." >&2
    stop_workers
    cleanup_manifests
    exit 130
}

trap cleanup_manifests EXIT
trap interrupted INT TERM

manifests=()
for ((worker = 0; worker < transfer_jobs; worker++)); do
    manifest="$manifest_dir/worker-$worker.files"
    : > "$manifest"
    manifests+=("$manifest")
done

worker=0
while IFS= read -r -d '' source_file; do
    relative_file=${source_file#"$local_dataset"/}
    printf '%s\0' "$relative_file" >> "${manifests[$worker]}"
    worker=$(( (worker + 1) % transfer_jobs ))
done < <(find "$local_dataset" -type f -print0)

echo "Syncing dataset with $transfer_jobs parallel rsync workers"
for ((worker = 0; worker < transfer_jobs; worker++)); do
    rsync "${rsync_args[@]}" \
        -e "$rsync_ssh" \
        --from0 \
        --files-from="${manifests[$worker]}" \
        "$local_dataset/" "$remote_host:$remote_dataset/" \
        > "$manifest_dir/worker-$worker.log" 2>&1 &
    worker_pids+=("$!")
    echo "Started worker $((worker + 1))/$transfer_jobs (PID ${worker_pids[$worker]})"
done

transfer_failed=0
for ((worker = 0; worker < transfer_jobs; worker++)); do
    if wait "${worker_pids[$worker]}"; then
        echo "Worker $((worker + 1))/$transfer_jobs completed"
    else
        echo "Worker $((worker + 1))/$transfer_jobs failed:" >&2
        tail -n 20 "$manifest_dir/worker-$worker.log" >&2
        transfer_failed=1
    fi
done
worker_pids=()

if [[ $transfer_failed -ne 0 ]]; then
    exit 1
fi

if [[ ${DRY_RUN:-0} == 1 ]]; then
    exit 0
fi

local_files=$(find "$local_dataset" -type f | wc -l)
local_bytes=$(find "$local_dataset" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f", total}')
remote_stats=$(ssh "${ssh_args[@]}" "$remote_host" \
    "find '$remote_dataset' -path '$remote_dataset/smoke32' -prune -o -type f -printf '%s\\n' | awk 'BEGIN {files=0; bytes=0} {files += 1; bytes += \$1} END {printf \"%d %.0f\\n\", files, bytes}'")
read -r remote_files remote_bytes <<< "$remote_stats"

if [[ "$local_files" != "$remote_files" || "$local_bytes" != "$remote_bytes" ]]; then
    echo "Dataset transfer verification failed." >&2
    echo "Local:  $local_files files, $local_bytes bytes" >&2
    echo "Remote: $remote_files files, $remote_bytes bytes" >&2
    exit 1
fi

echo "Dataset transfer verified: $remote_files files, $remote_bytes bytes"
