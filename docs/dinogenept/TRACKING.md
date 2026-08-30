# Hugging Face Trackio contract

DinoGenePT uses Trackio 0.37.0 to compare loss curves across datasets, seeds,
models, and ablation configs. Tracking is opt-in and fail-fast at run start: an
enabled experiment will not occupy a GPU if Trackio cannot initialize its
local durable database.

Tracker initialization is outside the scientific RNG contract: the runner
reseeds Python, NumPy, and Torch immediately afterwards. A regression test uses
a tracker stub that deliberately consumes all three RNG streams and requires
the final checkpoint to remain byte-identical to an untracked run.

The server records these epoch series:

- `train/loss` and individual prediction, DINO, local-DINO, delta-iBOT, KoLeo,
  and MoE diagnostics;
- `validation/prediction_mse` for DinoGenePT or `validation/loss` for Scouter;
- `optimizer/learning_rate`, epoch, batch count, and Trackio CPU/GPU telemetry;
- final `evaluation/<metric>/macro_mean` values and elapsed seconds.

Run names are deterministic:

```text
<model>--<dataset>--<experiment>--seed-<seed>--<config-hash-prefix>
```

The formal server environment is:

```bash
export DINOGENEPT_TRACKIO_ENABLED=true
export DINOGENEPT_TRACKIO_SITE=.runtime/trackio-site
export DINOGENEPT_TRACKIO_DIR=.runtime/trackio-data
```

After no run is writing the database, inspect the curves locally with:

```bash
TRACKIO_DIR=.runtime/trackio-data .venv/bin/trackio show \
  --project dinogenept-ablation
```

An isolated server integration check wrote and closed one epoch successfully;
Trackio reported the durable project directory and dashboard command. Formal
run names use the deterministic scheme above, so curves can be filtered by
model, dataset, ablation, seed, and resolved-config hash.

The compute server currently cannot reliably reach the Hugging Face API, so it
does not receive or persist a Hugging Face token. After runs have stopped,
checksum-sync `.runtime/trackio-data/` to the same ignored directory on the Mac
and upload from the authenticated Mac environment:

```bash
TRACKIO_DIR=.runtime/trackio-data .venv/bin/trackio sync \
  --project dinogenept-ablation \
  --space-id elan68681/dinogenept-ablation \
  --private
```

Do not sync a database while a server process is writing it. Never place an HF
or Ark token in a config, command argument, log, receipt, Git commit, or server
source tree. Trackio data is an observability mirror; formal comparisons and
reuse decisions read only the local hash-bound experiment receipts.

The installed Hugging Face connector is currently read-only and the Mac HF CLI
is not authenticated. Therefore the server-local dashboard data is ready for
loss-curve inspection, while publishing the private Space remains a separate,
explicitly authenticated Mac-side action.
