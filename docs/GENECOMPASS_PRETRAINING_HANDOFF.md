# GeneCompass human500k pretraining handoff

## Approved protocol

All500000 published cells train for one epoch. No validation split or best
validation checkpoint. Downstream overlap remains unknown: the release lacks
study/donor/original-cell IDs. Training losses are not heldout evidence.
Continuous published values are preserved; no raw-count relabeling or repeated
library normalization. Native vocabulary contains23113 human genes.

## Verified preparation

- Native510-shard conversion and lossless NPY migration passed full verification.
- Mmap manifest SHA256:
  `917a37a4a2b2a9bb2f927c9921a54b3b81c4b7ad213d9ce758b0068a84390f67`.
- IO comparison:512 identical random cells, seed42, two post-verification warm
  cache trials. NPZ155.62/155.68cells/s; NPY1417.97/1431.27cells/s.
  All crop hashes identical. This ~9x result concerns CPU reading/cropping,
  NOT GPU or end-to-end training speed. Server receipt:
  `results/genecompass-io-comparison-v1.json`.
- CPU-isolated server suite:180 tests passed; wheel/sdist build passed locally.
  Real default GPU smoke and training speed measurements remain outstanding.

## Launch gates and commands

Run only on the server in `/data/yilangliu/DinoGenePT`. Inspect current GPU
ownership and existing processes/output directories first. Do not reuse an
output directory without explicit checkpoint resume. Never preempt other work.
The following are prepared commands, NOT evidence that either run started.

```bash
.venv/bin/python scripts/launch_genecompass.py smoke
```

Confirm finite losses/gradients, one update, Teacher EMA and non-formal smoke
receipt/checkpoint before accepting the GPU execution stage. This preparation
stage only validates the launcher and CPU gate tests, not real GPU readiness.
Then, after smoke exits successfully:

```bash
.venv/bin/python scripts/launch_genecompass.py train
```

Run under a unique tmux session with durable logs and record its actual PID,
start identity, config hash and exit status. Check CLI/executable paths before
submission. Existing GPU guard must remain enabled. Full epoch acceptance:
completion receipt epoch=1, cells_seen=training_cells=500000, validation_cells=0,
expected optimizer steps, finite losses, checkpoint hash, performance records.
Use last.pt, not nonexistent best.pt, for this no-validation protocol.

## Monitoring

The old heartbeat was deleted. The newly authorized replacement must only be
registered after this preparation goal completes; see
[the monitor plan](GENECOMPASS_MONITOR_PLAN.md). No active goal and enabled
monitor overlap. Preserve active5M download; it is not a training prerequisite.
No5M training or perturbation fine-tuning. Training completion starts a scoped
validation phase; preparation completion alone is not overall completion.
