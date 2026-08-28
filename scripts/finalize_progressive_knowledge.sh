#!/usr/bin/env bash
set -euo pipefail

# Server-only materialization gate for the progressive knowledge experiment.
# The script is safe to leave in tmux: it waits for all checkpointed embedding
# writers, takes a non-blocking lock, and skips already materialized outputs.

project_root="/data/yilangliu/GenePT-Seed"
cd "$project_root"

python_bin=".venv/bin/python"
master_genes="data/universes/gradpert-ggi-master-genes.txt"
ggi_genes="data/ggi/genes-with-text.txt"
gradpert_root="/data/yilangliu/GraD-Pert/data-vnext-a942114"
model="doubao-embedding-vision"

conditions=(
  "protein"
  "protein-pathway"
  "protein-pathway-hpa"
)

mkdir -p results data/embeddings
exec 9>results/progressive-knowledge-finalize.lock
if ! flock -n 9; then
  echo "A progressive knowledge finalizer already holds the lock."
  exit 0
fi

while true; do
  ready=true
  for condition in "${conditions[@]}"; do
    raw="data/embeddings/seed-go-${condition}-master.npz"
    if [[ ! -s "$raw" ]]; then
      ready=false
    fi
  done
  if [[ "$ready" == true ]]; then
    break
  fi
  echo "$(date -Is) waiting for all three embedding NPZ files"
  sleep 60
done

for condition in "${conditions[@]}"; do
  corpus="data/corpora/seed-go-${condition}-master.json"
  checkpoint="checkpoints/doubao-seed-go-${condition}-master.sqlite3"
  checkpoint_audit="results/checkpoint-audit-seed-go-${condition}-master.json"
  raw="data/embeddings/seed-go-${condition}-master.npz"
  aligned="data/embeddings/seed-go-${condition}-master-aligned.npz"
  aligned_manifest="data/embeddings/seed-go-${condition}-master-aligned.manifest.json"

  "$python_bin" -m genept_seed audit-checkpoint \
    --texts "$corpus" \
    --checkpoint "$checkpoint" \
    --model "$model" \
    --genes "$master_genes" \
    --preserve-gene-case >"$checkpoint_audit"

  "$python_bin" - "$checkpoint_audit" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["requested"] == 17730, report
assert report["exact_cached"] == 17730, report
assert report["pending"] == 0, report
assert report["dimensions"] == [2048], report
PY

  if [[ ! -s "$aligned" || ! -s "$aligned_manifest" ]]; then
    "$python_bin" -m genept_seed data align-axis-vectors \
      --source "$raw" \
      --genes "$master_genes" \
      --output "$aligned" \
      --manifest "$aligned_manifest"
  fi
done

vector_audit="results/progressive-knowledge-vector-audit.json"
"$python_bin" -m genept_seed audit-knowledge-vectors \
  --genes "$master_genes" \
  --ggi-genes "$ggi_genes" \
  --gradpert-root "$gradpert_root" \
  --protein data/embeddings/seed-go-protein-master-aligned.npz \
  --pathway data/embeddings/seed-go-protein-pathway-master-aligned.npz \
  --hpa data/embeddings/seed-go-protein-pathway-hpa-master-aligned.npz \
  --expected-dimension 2048 \
  --output "$vector_audit"

for condition in "${conditions[@]}"; do
  result="results/ggi-genept-seed-go-${condition}-l2.json"
  if [[ ! -s "$result" ]]; then
    "$python_bin" -m genept_seed benchmark ggi \
      --vectors "data/embeddings/seed-go-${condition}-master-aligned.npz" \
      --output "$result" \
      --name "genept-seed-go-${condition}" \
      --normalize \
      --data data/ggi \
      --genes "$ggi_genes"
  fi
done

"$python_bin" -m genept_seed audit-ggi-comparison \
  --result results/ggi-latest-common-l2-final.json \
  --result results/ggi-genept-seed-extended-l2.json \
  --result results/ggi-genept-seed-extended-goexp-l2.json \
  --result results/ggi-genept-seed-go-protein-l2.json \
  --result results/ggi-genept-seed-go-protein-pathway-l2.json \
  --result results/ggi-genept-seed-go-protein-pathway-hpa-l2.json \
  --baseline genept-seed-extended-goexp \
  --output results/ggi-progressive-knowledge-comparison.json

echo "$(date -Is) progressive knowledge vector audit and GGI comparison complete"
