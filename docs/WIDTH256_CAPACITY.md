# Width256 capacity experiment — 2026-09-09

## Subsequent user selection: global batch108

The user selected54 cells/GPU x2 GPUs xaccumulation1 =108. The active recipe
now reflects this choice; below, microbatch4 and the global96 recommendation
describe the historical test handoff. Peak LR follows2e-4 *sqrt(108/1024)
=6.495190528e-5; warmup/decay settings are unchanged. No formal training or
monitor was launched. The recipe SHA below is the original pressure-test recipe
before this batch-only configuration change, not the updated file's hash.

User-requested change from the width768/depth4 balanced candidate. New50k
resolver uses `configs/cell/genecompass50k_width256_recipe.json`; the width768
recipe and historical checkpoints remain intact. No formal run is launched.

Only architecture differences from the balanced recipe:
- Backbone/gene width768 ->256.
- KDA heads6 ->2 (head dimension128 unchanged).
- MLA heads12 ->4 (head dimension64 unchanged).

Depth4 (3KDA+1MLA), query rank192, KV rank128, expression basis256, FFN
expansion1, shared DINO/iBOT projection1024/128/8192, crop and loss settings
remain unchanged. This is width-plus-head-count scaling, not a width-only
causal ablation. Student11,017,031 parameters; backbone8,384,967. Teacher is
a separate EMA copy excluded from these Student counts.

## Protocol and provenance

Same native `scripts/probe_genecompass_batch.py` as the width768 experiment:
two RTX5090, DDP, accumulation1, bf16 autocast/fp32 parameters, checkpointing.
Real GeneCompass50k cells, maximum-length views2048/2048/820/820, distinct cells
across ranks. Half of globals masked, concentrated in global0,20% shared
reconstruction/iBOT targets. Full losses, backward, clipping, optimizer and EMA.
Finite losses/gradients required on both ranks. Repeated batch and fixed probe
LR1e-6 measure capacity, not formal schedule stability, convergence or I/O speed.

- Recipe SHA256: `f218b4ca9dce3e85a17dd65e985bae432af8027fa51a9589296f1dbd32939ffd`.
- Probe SHA256: `7373fbc804031fcfe2a729daf61e19be9fa78b5aac70ca19e5b11d7c7d050f46`.
- Manifest SHA256: `73d490fc0e8fbfdb60f3827000d582daca15487ec80dc2438a1f53d3454d0d27`.
- Source data config SHA256: `945ad7f7036fdb0032b90b2dcb6de9275200b84ac96dba8044aadbdf8aae0768`.
- Server root: `/data/yilangliu/DinoGenePT/results/capacity/20260909-width256-b*`.
  Success requires `completion.json`; errors remain in corresponding
  `.runtime/capacity-width256-b*.log`. No weights overwritten/saved by probe.

## Results

| Cells/GPU | Global batch | Outcome |
|---:|---:|---|
|32|64|3 updates passed|
|48|96|10 updates passed; candidate with headroom|
|54|108|10 updates passed|
|55|110|Third update CUDA OOM after two complete updates|
|56|112|Second update CUDA OOM after first update passed|

Batch32/rank: peak allocated17,571,566,080 bytes, reserved22,573,744,128 bytes;
steady steps34.6 cells/s. Default allocator retained for comparison; fragmentation
can affect the boundary. A short pass is not a guarantee for a full epoch.

Batch48/rank: peak allocated26,195,935,744 bytes (24.40GiB),
reserved28,686,942,208 bytes (26.72GiB). Last-three-step throughput32.8–33.5
cells/s. Larger batch does not automatically improve throughput: batch32 was
about34.6 cells/s after startup. These short repeated-batch measurements are
not end-to-end training performance claims.

Batch54/rank: peak allocated29,429,989,376 bytes, reserved31,121,735,680 bytes;
last-three-step throughput35.6–36.7 cells/s.

Tested equal-rank boundary:54/rank, global108 under this exact configuration
and default allocator. Recommend48/rank, global96 for headroom; this does not
change the recipe's conservative microbatch4 placeholder or launch training.
Global256 is not supported by these measurements. Both GPUs were released at
the end (2MiB used each, no compute processes). No monitors were created.

Server CPU tests:25 passed (balanced/width256/tiny recipe, pretraining system,
GeneCompass launcher). Scoped Ruff and diff whitespace checks passed.
