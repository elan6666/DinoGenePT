# DINO mask and shared-head update — 2026-09-08

Status: new code defaults; previous training is stopped, not resumed or converted.

- Within each per-rank collated batch of B cells, randomly select exactly B of
  the 2B global views for masking. This does not force one global per cell.
- Selected globals hide floor(20% of valid genes), minimum one for tiny crops.
  The token-expression MSE, CLS-expression MSE and iBOT use identical positions.
  Unselected globals and locals have no reconstruction targets or hidden genes.
  Teacher always reads clean expression. Padding is never a target.
- Existing loss reduction is retained: zero-target cells contribute zero and
  remain in the per-cell denominator; globals are averaged for MSE/iBOT.
  Weights and the existing distillation ramp are unchanged.
- `heads.ibot_separate_head=false` shares the entire student DINO/iBOT projection
  module and its prototype weights. The teacher shares its own corresponding
  module; student and teacher do not share parameters. Cell/gene EMA centers
  remain separate. Default shared output dimension is 8192.
- `ibot_separate_head=true` restores independent heads: DINO 8192, iBOT 4096.
  `gene_prototypes` is inactive in shared mode.
- New recipe fields: `hidden_fraction=1.0`, `global_mask_probability=0.5`.
  Old frozen resolved configs/checkpoints must NOT be silently reused with these
  defaults. To reconstruct the old design explicitly set global probability 1,
  hidden fraction 0.8, and separate head true; retain old receipts unchanged.

Source references checked: official DINOv2 `dinov2/data/collate.py` (batch-global
selection), `dinov2/train/ssl_meta_arch.py` (shared projection, distinct loss
centering), and `dinov2/configs/ssl_default_config.yaml` (separate_head=false).
https://github.com/facebookresearch/dinov2

Adaptations: fixed 20% gene masks rather than 10–50% spatial block masks;
deterministic batch-ID/epoch seed; 8192 rather than 65536 default prototypes.
This document supersedes earlier current-default masking/head descriptions,
not historical experiment settings or results.
