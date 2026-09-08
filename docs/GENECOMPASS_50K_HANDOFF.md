# Official50k switch, September8

User stopped the500k dual-GPU run because of its duration and selected the
already downloaded official50k archive. Use all50000 published cells for ONE
fresh epoch. No weight transfer from500k, no validation, no5M training or
perturbation fine-tuning. Data overlap with downstream studies remains unknown.

## Preserved500k run

Verified torchrun1667217 received SIGTERM; ranks1667232/1667233 exited and
released both GPUs. Other job1456767 and downloader1768157 were left running.
The old output `results/pretraining/genecompass500k-mmap-one-epoch-v1` and logs
remain unchanged. Last observed optimizer record544; verified persistedlast.pt
contains525steps/67200cells. Steps after that checkpoint are not persisted model
updates. Queue and torchrun exit1 represent this requested interruption, not a
new training failure or successful epoch. Never automatically restart it.

## New data and method

- Official archive `data/official/genecompass-human/randsel_5w_human.tar.gz`:
  SHA256 `3d4ed25fa8dd48f616fd6267e9647fc55fcf1da2b0467d95e634a7db80c38f8b`.
- Content audit: `results/genecompass-50k-content-audit-v1.json`; two Arrow files,
 25000rows each, finite nonnegative continuous values, not discrete bins.
- `data/pretraining/genecompass-human50k-v1` contains50 native shards;
  lossless mmap version: `data/pretraining/genecompass-human50k-mmap-v1`.
- mmap manifest SHA256:
  `73d490fc0e8fbfdb60f3827000d582daca15487ec80dc2438a1f53d3454d0d27`.
- Frozen human23113-gene vocabulary, original published values and padding
  semantics retained. Values and IDs are checked; no new normalization.
- Backbone/heads/crops/training dictionaries were compared equal to the old
  resolved500k configuration.12layers/width768, same dualGPU/microbatch4/
  accumulation16/seed42; one epoch, scLong epoch-indexedLR1e-6 throughout.
  Total steps become391 (last window80cells). Progress-based loss/EMA schedules
  naturally use the new total; no manual hyperparameter retuning.

## Guarded execution

Fresh configs:
`.runtime/genecompass50k-mmap-smoke-config.json` and
`.runtime/genecompass50k-mmap-one-epoch-config.json`.
Formal configSHA256:
`945ad7f7036fdb0032b90b2dcb6de9275200b84ac96dba8044aadbdf8aae0768`.
The launcher checks explicit50k scope and refuses500k smoke reuse. Same
PID/start-ticks sharing policy and75% own PyTorch allocator cap; never preempt
another task. A changed smoke source/config requires revalidation.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/launch_genecompass.py smoke \
  --cells 50000 --share-with-process 1456767:419662683
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python scripts/launch_genecompass.py train \
  --cells 50000 --share-with-process 1456767:419662683
```

Commands are one-time fresh launches, not retry instructions. Smoke output:
`results/pretraining/genecompass50k-mmap-smoke-v1`; formal output:
`results/pretraining/genecompass50k-mmap-one-epoch-v1`. Each haslaunch/exit receipts.
tmux smoke `dinogenept-genecompass50k-smoke`, log `.runtime/genecompass50k-smoke.log`.
Formal launch intention: tmux `dinogenept-genecompass50k-train`, log
`.runtime/genecompass50k-train.log`. Inspect STATE for actual launch evidence.

## Handoff and monitoring

Finish preparation only after data, CPU tests and matching dualGPU smoke pass,
code/docs sync and guarded formal launch are ready. Then launch formal once,
briefly confirm real progress, and restore the SAME `dinogenept` heartbeat,
**every1hour, report every check**, covering50k training plus5M download.
Do not wait for the full epoch in goal mode. Old500k cancellation is expected.

Report steps/cells/percent, finite loss/gradient, LR, throughput/checkpoint and
download bytes/delta/speed/exit. Two unchanged checks warrant investigation;
explicit error is actionable immediately. No duplicate launch or automatic429
retries; respect Retry-After. Success requires formalexit0, epochs1,
training_cells=cells_seen50000, validation_cells0, completed_steps=total_steps391,
and validlast.pt; download additionally needs final size/gzipCRC/SHA.
