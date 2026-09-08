#!/usr/bin/env bash
# One explicitly authorized restart; archive lock is held by the downloader.
set -euo pipefail
cd /data/yilangliu/DinoGenePT
exec 9>.runtime/genecompass-download-slow60.lock
flock -n 9
test ! -e .runtime/genecompass-download-slow60.exit
test ! -e .runtime/genecompass-download-slow60.log
exec >.runtime/genecompass-download-slow60.log 2>&1
trap 'result=$?; echo "$result" > .runtime/genecompass-download-slow60.exit' EXIT
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
date -Is
.venv/bin/python -u scripts/download_genecompass_human.py \
  --workers 1 --bounded --request-interval 60
