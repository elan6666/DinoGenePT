# DINOv2-style LR — 2026-09-08

New default: `dinov2_step_cosine`; supersedes scLong LR for future resolved
GeneCompass runs. Existing immutable configs/checkpoints remain unchanged.
No training, batch probing, or monitoring was restarted.

Implementation references inspected directly:
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/configs/train/vitg14.yaml
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/config.py
- https://github.com/facebookresearch/dinov2/blob/main/dinov2/utils/utils.py

`training.learning_rate=2e-4` denotes BASE LR at batch 1024, not scaled peak.
Peak = base * sqrt(B/1024). B is microbatch * world_size * accumulation;
including accumulation is our extension (official loop does not accumulate).
Use the nominal configured B throughout, not the smaller last partial batch.
Crop multiplicity is not part of B.

Warmup steps W=floor(0.16*T), adapting 100000/625000 to our actual optimizer
step budget T. This is not a literal 100000-step warmup for small experiments.
Warmup follows numpy.linspace(0, peak, W), inclusive endpoints. Then:

    lr(s) = 1e-6 + (peak - 1e-6)/2 * (1 + cos(pi*(s-W)/(T-W)))

for zero-based W <= s < T. This matches the official denominator: the final
in-budget value approaches but is not exactly 1e-6; s >= T returns 1e-6.
No restarts. First LR is zero when W>0; W=0 skips warmup; W=1 has one zero
warmup value. Scheduler is a pure step function, stable on mid-epoch resume.

Examples: B=8 -> 1.76777e-5; B=16 -> 2.5e-5; B=128 -> 7.07107e-5;
B=3072 -> 3.46410e-4. For T=391: W=62. These are calculations, not new runs.

Scope: LR only. EMA remains existing 0.996-to-1; optimizer betas, weight decay,
teacher temperature and loss ramp are unchanged. Do not claim the entire
optimizer/teacher recipe is identical to DINOv2. Legacy scLong and cosine
implementations remain available for historical configurations.

Validation: 25 local CPU schedule tests including elementwise parity against
the official numpy formula, small budgets, batch scaling, and scLong regression.
