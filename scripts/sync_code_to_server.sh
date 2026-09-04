#!/usr/bin/env bash
set -euo pipefail

local_root="/Users/elan/code/DinoGenePT/"
remote="yilangliu@10.24.1.91:/data/yilangliu/DinoGenePT/"
ssh_key="/Users/elan/.ssh/elanquant_yilangliu_ed25519"
control_socket="/tmp/elanquant-agent.sock"

mode="--dry-run"
if [[ "${1:-}" == "--apply" ]]; then
  mode=""
elif [[ $# -gt 0 ]]; then
  echo "usage: $0 [--apply]" >&2
  exit 2
fi

rsync ${mode:+"$mode"} --checksum --recursive --delete --verbose \
  --exclude '.git/' \
  --exclude '/.venv/' \
  --exclude '/.runtime/' \
  --exclude '/.env' \
  --exclude '/uv.lock' \
  --exclude '/data/' \
  --exclude '/results/' \
  --exclude '/checkpoints/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '/dist/' \
  --exclude '/build/' \
  --rsh "ssh -i $ssh_key -S $control_socket -o IdentitiesOnly=yes -o BatchMode=yes" \
  "$local_root" "$remote"
