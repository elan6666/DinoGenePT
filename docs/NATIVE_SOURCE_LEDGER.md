# Native implementation ledger

Updated 2026-09-07. Reference reading is not a claim of whole-model parity.
No upstream model package is imported at runtime. Standard tensor and data
libraries remain permitted. Unit fixtures are not training results.

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

Server CPU verification: 8 distillation/LoRA tests, 5 KDA reference/chunk tests,
8 perturbation metric tests and 4 small-backbone tests passed. These are primitives, not a complete model
or a claim that the planned long-sequence training kernel meets performance goals.

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
