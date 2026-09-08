#!/usr/bin/env bash
# Single authorized campaign; fail on smoke, resource, identity or formal errors.
set -euo pipefail
cd /data/yilangliu/DinoGenePT
exec 9>.runtime/genecompass-shared-queue.lock
flock -n 9
test ! -e .runtime/genecompass-shared-queue.exit
test ! -e results/pretraining/genecompass500k-mmap-smoke-v1
test ! -e results/pretraining/genecompass500k-mmap-one-epoch-v1
trap 'result=$?; echo "$result" > .runtime/genecompass-shared-queue.exit' EXIT
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venv/bin/python scripts/launch_genecompass.py smoke --share-with-process 1456767:419662683
# The launcher independently checks matching successful smoke, metrics, source,
# checkpoint, exit and sharing identity before creating the formal output.
.venv/bin/python scripts/launch_genecompass.py train --share-with-process 1456767:419662683
