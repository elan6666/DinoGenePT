# DinoGenePT engineering, hardware and evaluation plan

Updated 2026-09-07. **Native cell modules and pretraining runner are CPU-tested;
formal data, GPU performance and experiment results remain unverified.**
Companion to [model design](CELL_DINO_KNOWLEDGE_LOCAL_DESIGN.md).

## 1. Native package and reference-first implementation

All cell models, losses, training and evaluators belong to `dinogenept`.
Do not import CellFM, Kimi, DINOv2, scGPT, GEARS, Scouter or GraD-Pert model
packages as runtime implementations. Standard infrastructure such as PyTorch,
CUDA/NCCL, NumPy and HDF5 remains appropriate. Preserve the existing GenePT
subsystem and artifact compatibility; do not restore retired model code blindly.

Before implementing each component:

1. Pin official source commit, files, symbols and license; identify unavailable
   code explicitly rather than inventing a supposedly official implementation.
2. Read called functions, shapes, masks, initialization, normalization, update
   order, defaults and optimizer behavior. A README is not enough.
3. Record source behavior -> native implementation -> declared deviations ->
   reference tests. Follow license/attribution obligations.
4. Independently implement the algorithm, then compare forward values,
   gradients, padding/masking, EMA/buffers and resume behavior on small fixtures.
5. Optimize only after the reference implementation passes. A native optimized
   KDA kernel must match the recurrence; do not replace it with an upstream
   model wrapper to claim native implementation.

Current native layout (extend through common interfaces for future models):

```text
src/dinogenept/
  cell/                      # current model, losses, runners, EMA/checkpoints
  datasets/<dataset_id>/      # dataset-specific native protocols
  evaluation/                # shared model-independent evaluator
  cli.py                     # lazy torch pretrain entrypoint + GenePT commands
  provenance.py              # shared checksum and atomic JSON primitives
configs/experiments/dinogenept/<dataset_id>/<variant>.json  # planned frozen configs
```

Different future models have their own folders but share the evaluator.
Every dataset/variant has a self-contained resolved config with provenance
labels (`official`, `user_locked`, `project_proposed`), not hidden inheritance.
The entrypoint runs preflight, initialization, fit, validation, sealed evaluation,
performance reporting and retention. No separate training script per ablation.

## 2. CellFM framework evidence

At commit `bfed59c0e34103231165d69b97927ecc888d623c`, CellFM uses MindSpore
Ascend `GRAPH_MODE`, explicit FP16 casting, dynamic loss scaling and data
parallelism; configuration enables recomputation of retention submodules.
The inspected path does not establish use of DeepSpeed or Lightning.

For our NVIDIA hardware, use native PyTorch DDP, mixed precision and activation
checkpointing. Selective compilation is an optimization candidate, not a
requirement to port MindSpore. The public training entrypoint contains undefined
`cut`, `args.datapath` and `args.savepath`, so do not claim it runs unchanged.
It actually constructs Adam with LR 1e-7, betas 0.9/0.95 and eps 1e-8;
config max/min LR values are not evidence of an active schedule. These are
reference values, not an experimentally selected optimizer for our new model.

Sources: [train.py](https://github.com/biomed-AI/CellFM/blob/bfed59c0e34103231165d69b97927ecc888d623c/train.py),
[config.py](https://github.com/biomed-AI/CellFM/blob/bfed59c0e34103231165d69b97927ecc888d623c/config.py),
[retention.py](https://github.com/biomed-AI/CellFM/blob/bfed59c0e34103231165d69b97927ecc888d623c/retention.py).

## 3. Two-5090 execution design

Read-only snapshot on 2026-09-07, not an idle reservation:

- Two RTX 5090 GPUs, each reporting 32,607 MiB total memory; driver 595.84.
- GPU interconnect is `SYS`, crossing NUMA nodes 0 and 3, not NVLink.
- CPU affinity reported GPU0: 0-15,64-79; GPU1: 48-63,112-127.
- Each card had a GraD-Pert process using about 6 GiB, about 26 GiB free.
- Host RAM about 251 GiB; `/data/yilangliu` about 1.2 TiB free.
- The existing GraD-Pert environment reported PyTorch 2.13.0+cu130, CUDA build
  13.0 and compiled sm_120 support. This is not proof of our environment,
  KDA kernel, BF16 numerical correctness or NCCL throughput.

Use a separate DinoGenePT environment; do not modify GraD-Pert's installation.
Recheck explicit GPU UUIDs, active jobs and capacity at launch. Authorization
to use two cards does not authorize killing existing processes. No training
or performance benchmark was run during the design inspection.

Default proposal: two-process DDP, each GPU holding a complete student and
no-grad EMA teacher. Reduce student gradients; update teacher locally after
synchronized optimizer steps. Reduce center sums/counts and synchronize routing
buffers according to their own contract. Do not default to one teacher GPU
and one student GPU, or treat two cards as one 64-GB memory pool.

| Setting | Initial capacity-test proposal, not measured fit |
|---|---|
| Model | 12 blocks, hidden width 768 |
| Per-rank microbatch | 8 original cells |
| Gradient accumulation | 8 steps |
| Effective batch | 2 x 8 x 8 = 128 original cells |
| Views | 2 globals + default 2 locals per original cell |
| Compute precision | BF16 autocast when verified; sensitive reductions FP32 |
| Parameters/optimizer/EMA | FP32 stability reference |
| Student activation checkpointing | Enabled with reproducible RNG behavior |
| Data workers | Start 2 per rank, bounded prefetch depth 2 |

Capacity probes try microbatches 4/8/16 with accumulation 16/8/4 respectively.
Include two teacher globals, student LC2/LC4/LC8, masked-token heads, backward,
optimizer state and communication buckets. Leave memory headroom. Do not infer
fit solely from parameter size. Run probes only when GPUs are available.

Use DDP no_sync on nonfinal accumulation steps. View-chunked backward must
preserve loss weights and perform one optimizer/EMA/schedule update per effective
step. Teacher targets are no-grad; do not retain needless teacher activations.
Test the DDP behavior of dynamic branches before claiming a static-graph fast
path. Do not silently alter optional-source sampling to avoid memory pressure.

KoLeo's neighborhood is the actual per-rank microbatch, not effective batch
128. Fix that size across primary comparisons and report any change; gradient
accumulation does not create more neighbors. Cross-rank neighbor gathering would
be a separately declared method change, not an automatic optimization.

Prefer DDP initially because SYS communication makes repeated cross-card model
sharding potentially expensive. Measure collective bandwidth before tuning
NCCL; topology alone does not prove P2P support/failure. Do not set workaround
flags without evidence. FSDP is a later capacity fallback, not the starting
assumption. No speedup factor is promised before measurements.

## 4. Entrypoint performance contract

Always emit compact periodic scalars and `performance_summary.json`:

- runtime/GPU identifiers, dtype and parameter counts (total/trainable/active);
- cold-start and compilation time separately from steady state;
- data wait, crop preparation, host-to-device, teacher/student forward,
  backward, optimizer/EMA, communication, validation, logging and checkpoint time;
- cells/s, effective steps/s, valid gene-tokens/s split by teacher/student
  and global/local, actual microbatch/accumulation and view lengths;
- per-rank peak allocated/reserved VRAM, CPU RSS, sampled utilization;
- p50/p95 step latency, profiling sample/window sizes and overhead;
- checkpoint, artifact and log bytes, available disk space.

Use CUDA events for GPU intervals: CPU enqueue time is not kernel duration.
Synchronize only bounded measurement windows, not every operator. Aggregate
DDP throughput using total work and slowest-rank elapsed time. Overlapped
communication/compute times are not additive. Keep profiler traces bounded.

Optimize batched sparse/HDF5 reads, bounded control caches, pinned host memory,
nonblocking copies, NUMA-local workers, length buckets, masked-token-only iBOT
head computation, checkpointing and knowledge-view chunks. Use supported SDPA
for the full-attention control; do not conflate MHA with Gated MLA. Compile only
after eager parity, recording dynamic-shape recompiles. Precision, sampling and
approximate kernels are explicit choices, not assumed semantics-preserving.

## 5. Perturbation data scope and GraD-Pert metric compatibility

Latest user update, 2026-09-07: pretraining uses an autonomously selected public
continuous-expression corpus (not the scGPT route), with two complete epochs;
the first perturbation campaign uses CellFM's Adamson and Norman datasets,
each with 10 LoRA fine-tuning epochs. No ablations are scheduled.
See `CELLFM_DATA_PROTOCOL.md` for verified sources and unresolved gates.
The following GraD-Pert protocol is a separate comparison contract, not an
instruction to replace CellFM splits or to launch all five datasets now.
Sharing metric formulas alone does not establish comparable results.

Read-only reference checkout `/Users/elan/code/grad-pert`, inspected HEAD
`276d7baac57c6a4130b5e43563cc7eccd8c8ff7e`:
`AGENTS.md`, `docs/design/DATA_AND_EVALUATION.md`,
`src/gradpert/evaluation/metrics.py`, `src/gradpert/training/checkpoint.py`,
`src/gradpert/execution/native.py`, `tests/execution/test_artifact_policy.py`,
and `configs/experiments/gradpert_b2/nadig_jurkat.yaml`.
Pin actual source and artifact hashes again before formal comparisons.

The previously audited GraD-Pert scope is five datasets: replogle_k562_essential,
replogle_rpe1_essential, nadig_jurkat, nadig_hepg2, norman. Older nine-dataset
discussion does not supply four additional audited protocols.

Consume frozen manifest data, not the gradpert runtime. Match canonical data,
preprocessing scale, gene order, condition eligibility, ordered split IDs,
control/context row IDs and evaluation-state hashes. Four single-target datasets
use the frozen PCG64 seed-42 split with target fractions 0.5625/0.1875/0.25.
Norman uses its predefined combo_seen2 split, not those fractions. Preserve the
shared GEARS representability intersection and retained order; no per-model
dropping or split regeneration. Equal seeds alone do not prove parity.
Run seeds 1/2/3/4 are replicates on the frozen split; seed 1 is the shared
comparison, not split seed 42 and not four independently generated splits.

Infer each condition on exactly its ordered 300 control draws from the frozen
context-matched manifest: Pred and InputCtrl each have shape [300,G]. Truth is
all real cells in that condition, not paired with prediction rows. Do not tile
one prediction mean 300 times. The following metrics are distinct:

| Metric ID | Exact per-condition definition |
|---|---|
| txpert_macro_pearson_delta | Pearson(mean(Pred)-mean(InputCtrl), mean(Truth)-mean(InputCtrl)) over all output genes |
| trishift_pearson_delta | Pearson of mean prediction/truth deltas from frozen MetricCtrlPoolMean, restricted to frozen DE_idx |
| systema_pearson | Pearson of mean prediction/truth minus the equal-condition-weight train+validation noncontrol centroid reference, restricted to frozen TopDE_idx |

Compute means/correlations in float64. Preserve the frozen DE method, target
exclusions, indices and availability reasons; no homemade Top-20 substitute.
Constant vectors, fewer than two genes and nonfinite inputs remain unavailable
with reasons, never coerced to zero. Macro-average available conditions and
report finite/total counts. Select checkpoints using validation-only
`val/txpert_macro_pearson_delta` and a fixed preregistered policy.

Evaluation-only DE/Truth and Systema references must not enter model training.
Preserve K562/Norman processed values and other datasets' audited preprocessing;
do not apply another CellFM normalization. MSE may be diagnostic. Accuracy,
AUROC and AP remain GenePT classification metrics, not this regression contract.

The native evaluator must match frozen fixtures on all three values, reasons,
denominators and hash/order checks. Include constant vectors, missing DE,
unequal population sizes, different reference pools and gene reorder failures.
Preregister numerical tolerance; float64 reference fixtures can target atol
1e-12. This fixture parity is required before reporting comparable results.

## 6. Compact artifacts

GraD-Pert's observed default is metrics_only: best.pt plus small metrics and
recipes, no persistent PKL. It removes last.pt after successful sealed
evaluation. It has an optional single-PKL export; ours remains disabled.

Proposed run layout:

```text
runs/<model>/<dataset>/<variant>/seed-<n>/<run_id>/
  checkpoints/best.pt
  checkpoints/last.pt          # interrupted/in-progress resume only
  small_results/resolved_config.json
  small_results/run_manifest.json
  small_results/inference_recipe.json
  small_results/metrics_per_condition.csv
  small_results/metrics_summary.json
  small_results/metric_availability.json
  small_results/performance_summary.json
  small_results/checkpoint_retention.json
  logs/scalars.jsonl
```

Use state dictionaries, not serialized whole model instances. Resume state
includes student/teacher, optimizer, centers, routing buffers, scheduler/scaler,
RNG/sampler progress and all identity hashes. Atomic writes precede hashing.
Keep last.pt after failure; remove redundant files only after durable success
receipts. A compact inference export and a full resume checkpoint have different
schemas/hashes; do not discard pretraining resume state without a declared policy.

`.pt` is not intrinsically smaller or pickle-free: torch.save serializes
metadata. Savings come from not duplicating Truth/control matrices or retaining
all predictions. Default evaluation streams per-condition aggregates, records
exact ordered IDs/seeds/checkpoint hashes, and retains no full per-cell matrices.
Optional requested exports use chunked numeric storage plus JSON, not automatic
giant PKLs. Existing GenePT NPZ artifacts are not mass-converted.

Large artifacts stay server-side. Sync only source/docs and reviewed small
receipts. No historical deletion is authorized by this plan. Cleanup is scoped
to exact reconstructible run files and recorded; no broad destructive globs.

## 7. Implementation order and acceptance

1. Source ledger and resolved configs: freeze modules, deviations, canonical
   manifests and environment; close undecided settings explicitly.
2. Native correctness: input/mask/zero tests, reference module parity,
   independent metric fixtures and complete checkpoint/resume tests.
3. Server capacity smoke: two available GPUs, full five-loss step, LC2/4/8,
   DDP gradient/EMA parity, measured memory/time and bounded profiler output.
4. One-epoch integration: chosen audited data, uninterrupted/resumed equivalence,
   frozen validation and metrics-only artifact checks; no claim of scientific
   improvement from this smoke.
5. Formal pretraining/fine-tuning and the active architecture/view/anchor
   ablations, identical evaluation contract and fixed loss weights. Dataset,
   epoch budget and source-fidelity gaps must be resolved before launching.

This plan is complete as a design deliverable. Capacity, CUDA-kernel parity,
new model implementation and scientific results are future verification work.
