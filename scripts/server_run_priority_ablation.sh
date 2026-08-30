#!/usr/bin/env bash
set -euo pipefail

root="/data/yilangliu/GenePT-Seed"
if [[ "$(pwd -P)" != "$root" ]]; then
  echo "run only from $root" >&2
  exit 2
fi

runtime="$root/.runtime"
lock="$runtime/locks/dinogenept-priority-ablation.lock"
log_root="$runtime/logs/dinogenept-priority-ablation"
mkdir -p "$runtime/locks" "$log_root" results/dinogenept
exec 8>"$lock"
if ! flock -n 8; then
  echo "priority ablation lock already exists: $lock" >&2
  exit 3
fi

export PYTHONPATH="src:.runtime/scouter-site"
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export DINOGENEPT_TRACKIO_SITE=".runtime/trackio-site"
export DINOGENEPT_TRACKIO_ENABLED=true
export DINOGENEPT_TRACKIO_PROJECT="dinogenept-ablation"
export DINOGENEPT_TRACKIO_DIR=".runtime/trackio-data"
export DINOGENEPT_TRACKIO_SYNC_SPACE_ID="elan68681/dinogenept-ablation"

python_bin="/data/yilangliu/GraD-Pert/source/.venv/bin/python"
gpu_index=0
gpu_uuid="GPU-23bb1466-b82b-982d-8eaf-9a0002c6a4e7"
gpu_wait_seconds="${DINOGENEPT_GPU_WAIT_SECONDS:-10800}"
gpu_lock_root="/data/yilangliu/.gpu-locks"
gpu_lock="$gpu_lock_root/${gpu_uuid}.lock"
mkdir -p "$gpu_lock_root"
observed_uuid="$(
  nvidia-smi --id="$gpu_index" --query-gpu=uuid --format=csv,noheader,nounits \
    | tr -d '[:space:]'
)"
if [[ "$observed_uuid" != "$gpu_uuid" ]]; then
  echo "physical GPU identity differs: expected=$gpu_uuid observed=$observed_uuid" >&2
  exit 4
fi

wait_for_files() {
  local missing
  while true; do
    missing=0
    for path in "$@"; do
      if [[ ! -s "$path" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      return
    fi
    sleep 15
  done
}

wait_for_gpu() {
  local free_mb utilization compute_pids idle_snapshots=0 started=$SECONDS
  while true; do
    free_mb="$(
      nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits \
        | tr -d '[:space:]'
    )"
    utilization="$(
      nvidia-smi --id="$gpu_index" --query-gpu=utilization.gpu \
        --format=csv,noheader,nounits | tr -d '[:space:]'
    )"
    compute_pids="$(
      nvidia-smi --id="$gpu_index" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | grep -E '^[[:space:]]*[0-9]+[[:space:]]*$' || true
    )"
    if [[ "$free_mb" =~ ^[0-9]+$ \
      && "$utilization" =~ ^[0-9]+$ \
      && -z "$compute_pids" ]] \
      && (( free_mb >= 30000 && utilization <= 5 )); then
      idle_snapshots=$((idle_snapshots + 1))
      if (( idle_snapshots >= 2 )); then
        return
      fi
    else
      idle_snapshots=0
    fi
    if (( SECONDS - started >= gpu_wait_seconds )); then
      echo "timed out waiting for exclusive idle GPU $gpu_uuid" >&2
      return 1
    fi
    sleep 5
  done
}

run_step() {
  local name="$1"
  shift
  local log="$log_root/${name}.log"
  local exit_file="$log_root/${name}.exit"
  if [[ -s "$exit_file" ]] && [[ "$(<"$exit_file")" == "0" ]]; then
    echo "$name already completed; revalidating through resumable command"
  fi
  exec 9>"$gpu_lock"
  if ! flock -w "$gpu_wait_seconds" 9; then
    echo "$name timed out acquiring shared GPU lock: $gpu_lock" >&2
    exit 5
  fi
  wait_for_gpu
  set +e
  "$@" >"$log" 2>&1
  local status=$?
  set -e
  printf '%s\n' "$status" >"$exit_file"
  flock -u 9
  if (( status != 0 )); then
    echo "$name failed; inspect $log" >&2
    exit "$status"
  fi
  echo "$name complete"
}

base="data/embeddings/dinogenept/base-targets.npz"
base_manifest="${base}.manifest.json"
wait_for_files "$base" "$base_manifest"

run_step supervised-norman-seed1 \
  "$python_bin" -m dinogenept run-pipeline \
  --pipeline configs/pipelines/ablation-00-supervised.yaml \
  --only norman --seed 1

run_step dino-base-norman-seed1 \
  "$python_bin" -m dinogenept run-pipeline \
  --pipeline configs/pipelines/ablation-01-dino-base.yaml \
  --only norman --seed 1

run_step dino-ibot-norman-seed1 \
  "$python_bin" -m dinogenept run-pipeline \
  --pipeline configs/pipelines/ablation-02-dino-ibot.yaml \
  --only norman --seed 1

run_step scouter-base-norman-seed1 \
  "$python_bin" -m dinogenept run-benchmark \
  --benchmark configs/benchmarks/scouter-gradpert5-base.yaml \
  --only norman --seed 1

go="data/embeddings/dinogenept/go-exp-only-targets.npz"
protein="data/embeddings/dinogenept/protein-only-targets.npz"
pathway="data/embeddings/dinogenept/pathway-only-targets.npz"
hpa="data/embeddings/dinogenept/hpa-only-targets.npz"
wait_for_files \
  "$go" "${go}.manifest.json" \
  "$protein" "${protein}.manifest.json" \
  "$pathway" "${pathway}.manifest.json" \
  "$hpa" "${hpa}.manifest.json"

run_step dynamic-locals-norman-seed1 \
  "$python_bin" -m dinogenept run-pipeline \
  --pipeline configs/pipelines/ablation-07-dynamic-locals.yaml \
  --only norman --seed 1

"$python_bin" - <<'PY'
import json
from pathlib import Path

from genept_seed.provenance import atomic_write_json, digest_file

root = Path("results/dinogenept")
paths = [
    root / "ablation-00-supervised.partial-67ac340b.json",
    root / "ablation-01-dino-base.partial-67ac340b.json",
    root / "ablation-02-dino-ibot.partial-67ac340b.json",
    root / "scouter-gradpert5-base.partial-67ac340b.json",
    root / "ablation-07-dynamic-locals.partial-67ac340b.json",
]
rows = []
receipts = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "partial_complete"
        or payload.get("row_count") != 1
        or payload["grid"].get("selected_datasets") != ["norman"]
        or payload["grid"].get("selected_seeds") != [1]
    ):
        raise ValueError(f"priority receipt grid differs: {path}")
    row = payload["rows"][0]
    rows.append(row)
    receipts.append({"path": str(path), "sha256": digest_file(path)})
fairness = {row["fairness_contract_sha256"] for row in rows}
metric_sets = {tuple(sorted((row.get("finetune") or row)["metrics"])) for row in rows}
if len(fairness) != 1 or len(metric_sets) != 1:
    raise ValueError("priority DinoGenePT/Scouter fairness contract differs")
result = {
    "schema_version": "dinogenept-priority-comparison-v1",
    "status": "partial_complete",
    "expected_grid": {"datasets": 5, "seeds": [1, 2, 3, 4]},
    "selected_grid": {"datasets": ["norman"], "seeds": [1]},
    "models": [
        "supervised",
        "dino-base",
        "dino-ibot",
        "scouter-base",
        "dynamic-locals",
    ],
    "fairness_contract_sha256": next(iter(fairness)),
    "metric_ids": list(next(iter(metric_sets))),
    "receipts": receipts,
    "rows": rows,
}
output = root / "priority-norman-seed1-comparison.json"
atomic_write_json(output, result)
atomic_write_json(Path(".runtime/dinogenept-priority-ablation.status.json"), result)
PY
