#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

"$script_dir/transfer_code_remote.sh" "$@"
"$script_dir/transfer_data_remote.sh" "$@"

