# Dual RTX 5090 batch capacity — 2026-09-09

## Result

Latest model, equal batch on both ranks, no gradient accumulation:

| Per GPU | Global batch | Complete updates | Result | Peak allocated / reserved per-rank maximum |
|---|---|---|---|---|
| 4 | 8 | 3 | pass | 22,377,986,048 / 25,014,829,056 bytes |
| 5 | 10 | 10 | pass | 27,561,349,632 / 30,490,492,928 bytes |
| 6 | 12 | 0 | CUDA OOM | allocation of another 234 MiB failed |

Observed tested limit is 5 per GPU (10 global), under this exact implementation
and allocator. This is a bounded capacity test, not proof of full-epoch stability
or a mathematical maximum under every allocator/input/kernel configuration.
Per-GPU 4 leaves more headroom. No formal run configuration was changed to 5.

## Conditions

- Both GPUs empty before launch; two RTX5090, about32GB each. No co-tenant killed.
- Real official GeneCompass50k cells, 23,113-gene vocabulary; distinct cells on
  both ranks selected with2048 valid genes. No synthetic padding shortcut.
- 12 layers,width768,batched_chunk KDA, activation checkpointing,bf16 autocast,
  fp32 parameters. Two2048-gene globals, two820-gene locals.
- Latest shared8192-prototype head,20% shared reconstruction/iBOT mask; exactly
  half of globals masked, concentrated in global0 to stress peak head memory.
- Fresh model, full losses at full ramp, AdamW lr1e-6 for capacity testing,
  betas(.9,.95), weight decay.01, gradient clip1, EMA update after every step.
  Fixed probe LR intentionally does not reproduce a formal LR trajectory.
- Repeated fixed real-cell stress batch; data-loader throughput not benchmarked.
  Batch5 warmed steps were approximately4.4–4.5s, about2.2–2.3 cells/s.
- Losses and clipped-input gradient norms finite on both ranks (fail-fast checks).
- Both ranks/processes released GPU memory after tests. No training or monitor
  restarted; old309-step training log/checkpoint preserved.

## Reproduction and receipts

Script: scripts/probe_genecompass_batch.py (native model, fresh output required).
Server outputs under results/capacity/20260909-shared-b{4,5,6}/; B4/B5 have
completion.json; B6 has started.json plus OOM log, never a success receipt.
Server logs: .runtime/capacity-shared-b{4,5,6}.log.
Recipe SHA256:4de58faf1c0bc9d937998e786ada663accbd2c41fc60548046349df3336ffc30.
Manifest SHA256:73d490fc0e8fbfdb60f3827000d582daca15487ec80dc2438a1f53d3454d0d27.
Model/crop identity and exact per-step memory/time are in completion receipts.

Launch from server repo with OMP_NUM_THREADS=4 MKL_NUM_THREADS=4:

```sh
.venv/bin/torchrun --standalone --nproc_per_node=2 scripts/probe_genecompass_batch.py \
  --config results/pretraining/genecompass50k-mmap-one-epoch-v1/resolved_config.json \
  --output results/capacity/NEW_UNIQUE_PATH --microbatch 5 --steps 10
```

The script takes data identity from the old config but explicitly overlays the
latest recipe's backbone (retaining gene count), heads, crops and optimizer.
It never loads or overwrites the formal checkpoint.
