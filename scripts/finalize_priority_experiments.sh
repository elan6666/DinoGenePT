#!/usr/bin/env bash
set -euo pipefail

cd /data/yilangliu/DinoGenePT
mkdir -p results/priority-1 results/priority-2

exec 9>results/priority-experiments-finalize.lock
if ! flock -n 9; then
  echo "priority experiment finalizer is already active" >&2
  exit 0
fi

conditions=(reactome signor masked shuffled)
declare -A corpora=(
  [reactome]=seed-go-protein-reactome-master
  [signor]=seed-go-protein-signor-master
  [masked]=seed-go-protein-pathway-signor-masked-master
  [shuffled]=seed-go-protein-pathway-signor-shuffled-master
)
declare -A labels=(
  [reactome]=seed-go-protein-reactome
  [signor]=seed-go-protein-signor
  [masked]=seed-go-protein-pathway-signor-masked
  [shuffled]=seed-go-protein-pathway-signor-shuffled
)

for condition in "${conditions[@]}"; do
  vector="data/embeddings/${corpora[$condition]}.npz"
  while [[ ! -s "$vector" ]]; do
    sleep 60
  done
  .venv/bin/python -m dinogenept audit-checkpoint \
    --preserve-gene-case \
    --texts "data/corpora/${corpora[$condition]}.json" \
    --genes data/universes/gradpert-ggi-master-genes.txt \
    --checkpoint "checkpoints/priority1-$condition.sqlite3" \
    > "results/priority-1/checkpoint-$condition.json"
  .venv/bin/python -m dinogenept audit-vectors \
    --vectors "$vector" \
    --genes data/universes/gradpert-ggi-master-genes.txt \
    --preserve-gene-case --expected-dimension 2048 --require-complete \
    > "results/priority-1/vector-$condition.json"
done

run_condition() {
  local condition="$1"
  local vector="data/embeddings/${corpora[$condition]}.npz"
  local label="${labels[$condition]}"

  if [[ ! -s "results/priority-1/fixed-$condition.json" ]]; then
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m dinogenept benchmark ggi \
      --vectors "$vector" --name "$label" \
      --data data/ggi --genes data/ggi/genes-with-text.txt --normalize \
      --output "results/priority-1/fixed-$condition.json" \
      > "results/priority-1/fixed-$condition.log" 2>&1
  fi
  if [[ ! -s "results/priority-1/gd-$condition.json" ]]; then
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m dinogenept benchmark ggi-gene-disjoint \
      --vectors "$vector" --name "$label" \
      --data data/ggi --genes data/ggi/genes-with-text.txt --normalize \
      --seeds 42,43,44,45,46,47,48,49,50,51 --test-fraction 0.2 \
      --output "results/priority-1/gd-$condition.json" \
      > "results/priority-1/gd-$condition.log" 2>&1
  fi
  if [[ ! -s "results/priority-2/properties-$condition.json" ]]; then
    OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -m dinogenept benchmark properties \
      --vectors "$vector" --name "$label" --normalize \
      --tasks data/properties/genept_property_tasks.csv \
      --genes data/properties/common-protein-latest-genes.txt \
      --folds 5 --seeds 42,43,44,45,46,47,48,49,50,51 \
      --output "results/priority-2/properties-$condition.json" \
      > "results/priority-2/properties-$condition.log" 2>&1
  fi
}

pids=()
for condition in "${conditions[@]}"; do
  run_condition "$condition" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

.venv/bin/python -m dinogenept audit-ggi-comparison \
  --result results/ggi-latest-common-l2-final-rerun.json \
  --result results/ggi-genept-seed-go-protein-l2.json \
  --result results/priority-1/fixed-reactome.json \
  --result results/priority-1/fixed-signor.json \
  --result results/ggi-genept-seed-go-protein-pathway-l2.json \
  --result results/priority-1/fixed-masked.json \
  --result results/priority-1/fixed-shuffled.json \
  --result results/ggi-genept-seed-go-protein-pathway-hpa-l2.json \
  --baseline genept-seed-go-protein \
  --output results/priority-1/fixed-comparison-final.json \
  > results/priority-1/fixed-comparison-final.stdout.json

.venv/bin/python -m dinogenept audit-gene-disjoint-comparison \
  --result results/priority-1/gd-latest.json \
  --result results/priority-1/gd-protein.json \
  --result results/priority-1/gd-reactome.json \
  --result results/priority-1/gd-signor.json \
  --result results/priority-1/gd-pathway.json \
  --result results/priority-1/gd-masked.json \
  --result results/priority-1/gd-shuffled.json \
  --result results/priority-1/gd-hpa.json \
  --output results/priority-1/gd-comparison-final.json \
  > results/priority-1/gd-comparison-final.stdout.json

.venv/bin/python -m dinogenept audit-property-comparison \
  --result results/priority-2/properties-latest.json \
  --result results/priority-2/properties-protein.json \
  --result results/priority-2/properties-reactome.json \
  --result results/priority-2/properties-signor.json \
  --result results/priority-2/properties-pathway.json \
  --result results/priority-2/properties-masked.json \
  --result results/priority-2/properties-shuffled.json \
  --result results/priority-2/properties-hpa.json \
  --output results/priority-2/property-comparison-final.json \
  > results/priority-2/property-comparison-final.stdout.json

printf 'complete\n' > results/priority-experiments-finalize.complete
