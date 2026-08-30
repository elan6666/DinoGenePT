#!/usr/bin/env bash
set -euo pipefail

root="/data/yilangliu/GenePT-Seed"
if [[ "$(pwd -P)" != "$root" ]]; then
  echo "run only from $root" >&2
  exit 2
fi
if [[ -z "${ARK_API_KEY:-}" ]]; then
  echo "ARK_API_KEY is absent; inject it privately in this shell" >&2
  exit 3
fi

runtime="$root/.runtime"
lock="$runtime/locks/dinogenept-prior-generation.lock"
mkdir -p "$runtime/locks" "$runtime/logs" checkpoints data/embeddings/dinogenept
if ! mkdir "$lock" 2>/dev/null; then
  echo "prior generation lock already exists: $lock" >&2
  exit 4
fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT

genes="data/universes/gradpert-five-targets.txt"
model="doubao-embedding-vision"
pids=()
names=()

start_embedding() {
  local name="$1"
  local texts="$2"
  local checkpoint="$3"
  local output="$4"
  local profile="$5"
  local log="$runtime/logs/${name}.log"
  if [[ -s "$output" && -s "${output}.manifest.json" ]]; then
    echo "$name already materialized; skipping"
    return
  fi
  (
    PYTHONPATH=src .venv/bin/python -m genept_seed embed \
      --texts "$texts" \
      --genes "$genes" \
      --preserve-gene-case \
      --checkpoint "$checkpoint" \
      --output "$output" \
      --model "$model" \
      --expected-dimension 2048 \
      --batch-size 10 \
      --max-workers 1 \
      --request-interval 20 \
      --profile "$profile"
  ) >"$log" 2>&1 &
  pids+=("$!")
  names+=("$name")
  echo "$name started pid=$!"
}

start_embedding \
  base \
  data/corpora/seed-master.json \
  checkpoints/dinogenept-base-targets.sqlite3 \
  data/embeddings/dinogenept/base-targets.npz \
  ncbi-uniprot-base-v1
sleep 4
start_embedding \
  go \
  data/corpora/dinogenept/go-exp-only-targets.json \
  checkpoints/dinogenept-go-exp-only-targets.sqlite3 \
  data/embeddings/dinogenept/go-exp-only-targets.npz \
  go-exp-only-v1
sleep 4
start_embedding \
  protein \
  data/corpora/dinogenept/protein-only-targets.json \
  checkpoints/dinogenept-protein-only-targets.sqlite3 \
  data/embeddings/dinogenept/protein-only-targets.npz \
  interpro-uniprot-structure-only-v1
sleep 4
start_embedding \
  pathway \
  data/corpora/dinogenept/pathway-only-targets.json \
  checkpoints/dinogenept-pathway-only-targets.sqlite3 \
  data/embeddings/dinogenept/pathway-only-targets.npz \
  reactome-signor-only-v1
sleep 4
start_embedding \
  hpa \
  data/corpora/dinogenept/hpa-only-targets.json \
  checkpoints/dinogenept-hpa-only-targets.sqlite3 \
  data/embeddings/dinogenept/hpa-only-targets.npz \
  hpa-only-v1

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "${names[$index]} complete"
  else
    echo "${names[$index]} failed; inspect its bounded log" >&2
    failed=1
  fi
done
exit "$failed"
