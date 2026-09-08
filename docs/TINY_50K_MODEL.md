# 50k-cell / approximately 500k-parameter model

User requested parameter/cell ratio approximately10. We count unique Student
parameters INCLUDING pretraining heads, excluding the non-gradient EMA copy.

Current experiment recipe: configs/cell/genecompass50k_500k_recipe.json.
The existing census500k_default_recipe.json remains the historical large
model recipe; its 82.85M architecture and checkpoints are not overwritten.
The resolver selects this recipe for NEW50k configs by default; an explicit
--reference-config overrides it. Existing frozen paths must not be reused.

## Size

- Student495570 parameters:9.9114 per cell,0.886% below500000 target.
- Backbone419450; pretraining heads/reconstruction76120.
- Teacher495570 additional parameters; no independent backpropagation.
- Keep23113 genes,12 layers (9 KDA +3 MLA), 2global+2local,2048 gene cap,
  shared8192 prototypes,20% shared iBOT/MSE masks,half batch globals masked.
- Width16;1 KDA head of16;1 MLA head of16;shared8,query/kv ranks8.
- Expression basis128 (large model default remains256).
- Shared projection hidden32,bottleneck8. Full losses/EMA unchanged.
- Tiny recipe uses accumulation1. No formal training launched.

This is a deliberately constrained small-model experiment, not a claimed
optimal scaling law. Gene identity table alone contains369824 parameters.
Width16 and bottleneck8 may limit biological representation quality; evaluate
downstream before drawing conclusions. This cannot reuse large-model weights.

## Capacity protocol

Use scripts/probe_genecompass_batch.py --recipe configs/cell/genecompass50k_500k_recipe.json.
Both GPUs initially unoccupied. Real distinct GeneCompass cells, maximum crops
[2048,2048,820,820], worst-case concentration of half-batch masks in global0.
BF16 autocast/FP32 weights, checkpointing, full AdamW+EMA updates, accumulation1.
Probe LR fixed1e-6 (capacity test, not the formal DINOv2 LR trajectory).
Outputs results/capacity/20260909-tiny-b*/; logs .runtime/capacity-tiny-b*.log.

Model parameter count affects model/gradient/optimizer memory. Batch capacity
also depends on activation tensors, token length, expression basis, prototype
logits and attention/kernel temporary allocations. It does not grow inversely
with parameter count. In particular,8192-way token logits remain expensive.

CPU validation: tiny-size budget, configurable expression basis and legacy
default, full reduced-model backward, DINO/iBOT shared/independent heads and EMA.

## Verified dual-5090 stress results (no accumulation)

| Per rank | Global batch | Complete updates | Result | Peak allocated bytes (max rank) |
|---|---|---|---|---|
| 32 | 64 | 3 | pass | 5664348160 |
| 160 | 320 | 3 | pass | 27402400256 |
| 184 | 368 | 3 | pass | 31341049344 |
| 190 | 380 | 10 | pass | 32344545792 |
| 191 | 382 | 1 | OOM during next update | 32485681664 before failure |
| 192 | 384 | 0 | OOM | no completed update |

Tested boundary190 per GPU; long-run stability is not asserted. Recommend
160 per GPU for headroom (not a formal launch or full-epoch guarantee).
Follow-up user choice: recipe microbatch128, two GPUs, accumulation1 = global
batch256. Scaled peak LR1e-4. For50000 cells and1epoch:196 updates,31 warmup
updates (floor16%). This exact batch was not separately pressure-tested;
larger batches above passed. Pressure receipts retain their original recipe
hash; choose a fresh resolved run config before any formal launch.
B190 warmed updates ~4.6–4.7s,~81–82 cells/s, allocated30.12GiB,
reserved30.28GiB. B160 allocated25.52GiB. Timings exclude loader and use
repeated fixed cells, so are not end-to-end epoch speed guarantees.
Loss/gradient finite checks passed both ranks for successful runs. All probe
processes exited and GPUs released.25 CPU regression tests passed, Ruff passed.

Recipe SHA256 e441c6bd46d7dc50982c8e98eb6ac1bfa0923125ebed68b0ec1b1454cd11f9af.
Completed JSON receipts in server results/capacity/20260909-tiny-b32,
b160,b184,b190. B191/B192 contain failure evidence only, never success receipts.
