#!/usr/bin/env bash
set -euo pipefail


local_root="/home/minh-le-vo-nhat/Documents/Minh-DUT/NCKH/NewAttempt2627/high-resolution-viton-zalando-dataset"
remote_root="/home/azr-ai/Documents/archive"
remote_host=${3:-azr-ai@100.75.140.87}

for path in "$local_root" "$remote_root"; do
    if [[ "$path" =~ [[:space:]] ]]; then
        echo "Dataset paths containing whitespace are not supported: $path"
        exit 2
    fi
done

for split in train test; do
    mask_dir="$local_root/$split/agnostic-mask"
    if [[ ! -d "$mask_dir" ]]; then
        echo "Missing local mask directory: $mask_dir"
        exit 1
    fi
done

shopt -s nullglob
pair_files=("$local_root"/*pairs*.txt)
if (( ${#pair_files[@]} == 0 )); then
    echo "No pair files matching '*pairs*.txt' under $local_root"
    exit 1
fi

rsync_args=(-avh --info=progress2 --partial)
if [[ ${DRY_RUN:-0} == 1 ]]; then
    rsync_args+=(--dry-run)
    echo "Dry run: remote directories will not be created"
else
    ssh "$remote_host" mkdir -p \
        "$remote_root/train/agnostic-mask" \
        "$remote_root/test/agnostic-mask"
fi

rsync "${rsync_args[@]}" "${pair_files[@]}" "$remote_host:$remote_root/"
for split in train test; do
    rsync "${rsync_args[@]}" \
        "$local_root/$split/agnostic-mask/" \
        "$remote_host:$remote_root/$split/agnostic-mask/"
done

echo "Transferred pair lists and agnostic masks to $remote_host:$remote_root"
