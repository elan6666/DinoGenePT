# Native implementation ledger

Updated 2026-09-12. Reference reading is not a claim of whole-model parity.
No upstream model package is imported at runtime. Standard tensor and data
libraries remain permitted. Unit fixtures are not training results.

## 2026-09-12 ablation implementation sources

- Muon: KellerJordan/Muon `f98f1cacc0263b04290753e32be8d498c1efc806`,
  `muon.py`, NS polynomial/Nesterov/auxiliary Adam. Independent native code;
  FP32 NS instead of reference BF16, explicit model-aware groups, no redundant
  optimizer collectives, batched equal-sized heads. GLM/Kimi use project RMS
  scale0.2; DS0.18. No claim of bitwise parity with three proprietary trainers.
- GLM5: arXiv `2602.15763v1` §2.1/AppendixA: Q/K/V up-projection split and LR.
- KimiK3: arXiv `2607.24653v1` §2.5/3.3. Weight-clip formula inspected in
  KimiK2 `2507.20534v1` §2.1 Algorithm1, tau100; MLA shared-channel rule retained
  with NoPE adaptation. Fixed NS/Nesterov implementation is shared to isolate layout.
- DS4.1: official `deepseek-ai/DeepSeek-V4.1-Flash/DeepSeek_V41_Tech_Report.pdf`,
  §2.5/Algorithm1/§4.2.2 inspected: Q/K head split, Sinkhorn and LR phases.
  Fixed-budget warmups/step ratios and conservative auxiliary groups are ours.
- CellFM `bfed59c0e34103231165d69b97927ecc888d623c`, `retention.py` and
  `attention.py` read: native ERet mixer ReLU Q/K, Q(K^T V), scale-free inner
  RMS, SiLU U gate; optional bilinear SGLU and post-LN DeepNorm-form residual.
  Shared model initialization/output norm, masks and gene crops are our choices;
  this is not a whole-model CellFM reproduction or a copied MindSpore module.
- Gated Delta: FLA `9d981ffef3b361ba931102b633ae2a9fd91ca6c3`,
  `fla/ops/gated_delta_rule/naive.py`. Scalar log decay broadcast into our KDA
  state equation is algebraically the same delta update (same query scaling).
- Qwen code reference: Transformers `df04b012229d50d2b6dfba32c61c3057c3a40ea1`,
  `models/qwen3_5/modeling_qwen3_5.py` recurrent/chunk delta and gate mapping;
  released Qwen3.8-27B config identifies qwen3_5 model classes. Our reduced
  variant has equal K/V head counts, NoPE, noncausal grouped global attention,
  shared native init, optional convolution; not a full Qwen3.8 implementation.
- iBOT chunking is our checkpointed algebraic implementation, not a claimed
  imported DINO/GLM kernel. Same per-cell weights and one sum/count center update.

See `ABLATION_IMPLEMENTATION_20260912.md` for current coverage and short-test
limits; campaign descriptions below are historical.

| Mechanism | Inspected reference | Native implementation / deviations |
|---|---|---|
| LoRA linear | microsoft/LoRA `c4593f060e6a368d7bb5af5273b8e42810cdef90`, loralib/layers.py | cell/lora.py; same alpha/r and A-random/B-zero linear update; freezes bias too; never merges in-place on eval; explicit exact paths |
| DINO head and centers | facebookresearch/dinov2 `7764ea0f912e53c92e82eb78a2a1631e92725fc8`, layers/dino_head.py, loss/dino_clstoken_loss.py | cell/distillation.py; native PyTorch weight parametrization; explicit synchronous sum/count DDP center updates; call after forming targets |
| iBOT | same DINOv2 revision, loss/ibot_patch_loss.py | native per-cell masked CE, caller must align genes within each global; centers count valid tokens rather than padding |
| KoLeo | same revision, loss/koleo_loss.py | native FP32, same PairwiseDistance epsilon convention; diagonal -inf prevents selecting self in antipodal ties |
| Reconstruction | CellFM `bfed59c0e34103231165d69b97927ecc888d623c`, model.py, loss_function.py | per-cell masked mean is our variable-crop adaptation; upstream pools selected tokens globally |
| KDA / Gated MLA | moonshotai/Kimi-K3 HF revision `f831ab66814297da540d832a5235f8e904f29d06`, modeling_kimi_linear.py | mixer module audit in progress; no-convolution retains SiLU activation by explicit adaptation |
| KDA state equation | fla-org/flash-linear-attention `9d981ffef3b361ba931102b633ae2a9fd91ca6c3`, fla/ops/kda/naive.py and gate.py | cell/kda.py: native recurrence plus independently expressed triangular-solve chunks; CPU values/gradients agree, fused CUDA parity and throughput pending |

DINO cross-global/local loss pair averaging and same-global exclusion are
project orchestration choices, not the Cartesian-sum helper copied verbatim.
Teacher parameter update requires identical parameter/buffer names and shapes.
Teacher never receives gradients. Separate cell/token centers must be checkpointed.

Kimi's configuration lives under `text_config`, not the top-level multimodal
config. Verified KDA config: head_dim128, bounded log gate -5, full-rank sigmoid
output gate, conv kernel4 (our default omits convolution). Gated MLA retains a
shared extra key channel even with NoPE and no rotary operation. Reduced model
dimensions need explicit mapping; do not discard these channels as a shortcut.

Server CPU verification includes primitive parity, full five-loss orchestration,
two-rank Gloo accumulation and exact interrupted/resumed optimizer-boundary
continuation. These fixtures do not establish CUDA capacity or formal training.
CUDA capacity has now been measured separately; see
`CUDA_CAPACITY_2026_09_07.md`. Original native chunk remains the default: a faster
algebraically equivalent scheduling candidate passed FP32 tests but failed the
whole-backbone BF16 elementwise acceptance gate. No fused-kernel parity claim.
Follow-up: order-preserving `batched_chunk` passes full2048-token/60664-vocabulary
FP32/BF16 forward audits and long-sequence CUDA gradients, with matched single-
card performance evidence. Matching two-rank capacity is still needed before
the proposed formal recipe can launch. See `DEFAULT_PRETRAINING_LAUNCH.md`.

## Reduced default backbone now implemented

`cell/backbone.py`: 12/768, 6 KDA heads x128; global layer every fourth block;
MLA 12 heads, 64 private QK/value channels +32 shared QK channels, query rank192,
KV rank128; no rotary encoding, noncausal global visibility. Block AttnRes spans
4 backbone blocks; dense width-1 SiTU-GLU (4,25), no MoE in this default.
Source-compatible mechanisms have the reduced dimensions explicitly declared.

From-scratch adaptations: normal std0.02 linear/gene initialization, zero initial
depth queries, and dt_bias chosen for initial log decay -0.1 at zero projection.
HF inference source allocates dt_bias with `empty` and expects loaded weights;
copying that allocation into a from-scratch model would be unsafe. This is our
declared initializer, not a claimed reproduction of Kimi's training initializer.
CellFM value/head formulas are retained, with explicit masking and a reduced
gene embedding table. No implicit PAD/zero conflation; padded states are zeroed.

LoRA target paths are enumerated by module type: KDA q/k/v/o; MLA q_a/q_b/kv_a/
kv_b/o. Gate/decay, gene/value encoders, norms and residual parameters freeze
during backbone LoRA tuning. New task/source heads remain outside the backbone.

## Frozen campaign scope

- Default only, no ablations: 2 complete pretraining epochs and 10 LoRA
  fine-tuning epochs on each of Adamson and Norman.
- Pretraining trains full weights; fine-tuning freezes pretrained base weights.
- LoRA initial proposal rank16/alpha32/dropout0.05 on explicitly audited mixer
  projections. Target paths will be fixed after backbone implementation;
  do not insert into every linear by substring matching.
- New prediction/source projection heads train normally. Knowledge vectors freeze.
- DDP capacity must be measured on unoccupied GPUs before formal training.

## Native orchestration and data protocol integration

- `cell/sampling.py`: CellFM weighted log1p-count cap without replacement,
  full-library normalize/log1p before cap, independent ratio crops. Epoch/cell
  keyed randomness; no dropped/repeated cells in balanced DDP batches. Our tiny
  crop guard selects at least one reconstruction/hidden target rather than
  propagating the upstream zero-length slicing corner case. Targets are 20%
  and hidden positions 80% of targets, with this explicitly declared guard.
- `cell/pretraining.py`: clean two-global Teacher, masked globals and clean
  locals for Student; hidden-position-only prototype head; two per-cell MSEs,
  paired DINO, same-global aligned iBOT, sum of two backbone-CLS KoLeo terms.
- `cell/train.py` / checkpoint: native AdamW/DDP/BF16 entrypoint, all-cell epoch
  accounting, EMA after optimizer, separate loader RNG and atomic weights-only
  loadable checkpoints. CPU exact resume and two-rank tests passed. First-block
  depth-read query/norm removed: no history exists there and unused parameters
  caused a real DDP reducer failure. No numerical operation was removed.
- `cell/perturbation.py`: our population-bag adaptation, not unavailable
  DINOcell source code. LoRA on frozen backbone, reused trainable cell projection
  head, new per-source adapters and factorized full-axis delta decoder. Missing
  source omitted; all targets required for combinations; main-only prediction
  API cannot consume post-treatment observations. See model design for formulas.
- `datasets/cellfm/split.py`: fixed `simulation` protocol only. Legacy MT19937
  and upstream reseeding preserved without mutating global NumPy RNG. Validation
  re-splits training with fractions 0.9/0.9 and the same seed. Tests execute only
  the pinned reference class/helper AST, never import GEARS models. Exact member
  and subgroup parity passed on single/combo fixtures for seeds 1,3,42.
  Reference hashes: data_utils.py
  `776b41af9c8d49131d103ef54c8e0bea892a8313137e9c191fd8e74a41c05b58`;
  utils.py `d24d35a50741b011e5e8078d2666aa0d1dfea535a98101691d523ab00e419a9a`.
  Released Adamson/Norman input files are now extracted and source/hash checked;
  all delivered target/output genes map exactly to the frozen Census vocabulary.
  Split reference parity above is fixture-level; no claim of full GraD-Pert data
  or proven norman-1000 notebook-file identity.
- `datasets/cellfm/dataset.py`: own frozen sparse-data/bag interface; AnnData's
  public read_elem is used solely for X/obs/var encoded storage, not models or
  evaluation-only uns annotations. Real dataset IO coverage receipt recorded.
- `cell/transfer.py`: own weights-only checkpoint lineage gate, exact native
  Student transfer; completed two-epoch formal run required before fine-tuning.
- `evaluation/cellfm.py`: read-only GraD-Pert276d7ba controls/state/metrics method
  reference; own300-control PCG64 draw, t-test DE eligibility/target exclusion
  and Systema reference. Frozen real evaluation states and limitations are in
  `CELLFM_LORA_PROTOCOL.md`; data/split identity is NOT GraD-Pert canonical.
- `cell/finetune.py`: own ten-complete-epoch LoRA loop, validation-only selection,
  test after training, immutable input checks, checkpoint/RNG resume and scalar
  performance receipts. Small CPU end-to-end/resumption tests pass, not formal
  CUDA training results. Native optimizer defaults are explicitly adaptations.
