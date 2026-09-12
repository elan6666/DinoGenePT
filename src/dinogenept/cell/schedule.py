"""Native epoch-indexed LR, matching scLong's sequential scheduler.step().

Reference: BaiDing1234/scLong@41b72021540918e4386c4f2264351d7a6cbeed99,
utils.py / pretrain_dual_4096_all_1b_mix.py. No upstream runtime imports.
"""

import math


def teacher_momentum(completed_steps: int, total_steps: int, initial: float = 0.994) -> float:
    """DINOv2 zero-based iteration schedule, applied after optimizer.step()."""
    if not 1 <= completed_steps <= total_steps or not 0 <= initial < 1:
        raise ValueError("Invalid completed optimizer step count")
    return 1 - (1 - initial) * (1 + math.cos(math.pi * (completed_steps - 1) / total_steps)) / 2


def recipe_learning_rate(name, base_lr, *, batch, step, total_steps, warmup_fraction=None,
                         min_lr=None):
    """Small-model adaptations, not the upstream token budgets or absolute LR.

    GLM5: cosine to 20% peak; Kimi K3: 1% warmup/cosine; DS4.1:
    constant -> cosine at 28/45 budget -> floor at 40/45 budget.
    Project presets round DS boundaries to .62/.89. Explicit overrides permit
    controlled warmup/floor comparisons. All schedules index successful steps.
    """
    if name not in {"glm5_cosine", "kimi3_cosine", "ds41_plateau_cosine"}:
        raise ValueError("Unknown recipe schedule")
    if total_steps < 1 or step < 0 or batch < 1 or not math.isfinite(base_lr) or base_lr <= 0:
        raise ValueError("Invalid recipe schedule dimensions")
    peak = base_lr * math.sqrt(batch / 1024)
    warm = (0.01 if name == "kimi3_cosine" else 0.05) if warmup_fraction is None else warmup_fraction
    floor = ({"glm5_cosine": peak * .2, "kimi3_cosine": 1e-6,
              "ds41_plateau_cosine": peak * .1}[name] if min_lr is None else min_lr)
    if not 0 <= warm < 1 or not math.isfinite(floor) or not 0 <= floor <= peak:
        raise ValueError("Invalid warmup/floor")
    if name == "ds41_plateau_cosine" and warm >= .62:
        raise ValueError("DS warmup must finish before plateau ends")
    if step >= total_steps:
        return floor
    nw = int(total_steps * warm)
    if step < nw:
        return peak * step / max(1, nw - 1)
    start, end = ((.62 * total_steps, .89 * total_steps) if name == "ds41_plateau_cosine"
                  else (nw, total_steps))
    u = min(1., max(0., (step - start) / (end - start)))
    return floor + (peak - floor) * (1 + math.cos(math.pi * u)) / 2


def sclong_learning_rate(epoch: int, *, max_lr: float = 5e-5) -> float:
    """LR used during zero-based epoch; rebuildable exactly on mid-epoch resume.

    Warmup remains five epochs. Only the cosine portion doubles on restart.
    This follows step() with NO epoch argument, not the upstream's different
    explicit-epoch shortcut. No interpolation within an epoch is performed.
    """
    if type(epoch) is not int or epoch < 0:
        raise ValueError("Epoch must be a nonnegative integer")
    if not math.isfinite(max_lr) or max_lr < 1e-6:
        raise ValueError("Peak LR must be finite and at least 1e-6")
    position, length, cycle = epoch, 15, 0
    while position >= length:
        position -= length
        length = (length - 5) * 2 + 5
        cycle += 1
    peak = max_lr * 0.9**cycle
    if position < 5:
        return 1e-6 + (peak - 1e-6) * position / 5
    return 1e-6 + (peak - 1e-6) * (1 + math.cos(math.pi * (position - 5) / (length - 5))) / 2


def dinov2_learning_rate(base_lr: float, *, batch: int, step: int, total_steps: int,
                        warmup_fraction: float = 0.16, min_lr: float = 1e-6) -> float:
    """Official sqrt_wrt_1024 and CosineScheduler indexing; no cycle restarts.

    Adaptation: floor(0.16 * total_steps) warmup instead of fixed 100k/625k.
    Batch includes accumulation in our runner; upstream does not accumulate.
    """
    if (batch < 1 or total_steps < 1 or step < 0 or not 0 <= warmup_fraction < 1
            or not math.isfinite(base_lr) or base_lr <= 0
            or not math.isfinite(min_lr) or min_lr < 0):
        raise ValueError("Invalid DINOv2 schedule")
    peak = base_lr * math.sqrt(batch / 1024)
    if min_lr > peak:
        raise ValueError("Minimum LR exceeds scaled peak")
    if step >= total_steps:
        return min_lr
    warmup = int(total_steps * warmup_fraction)
    if step < warmup:
        return peak * step / max(1, warmup - 1)
    return min_lr + 0.5 * (peak - min_lr) * (
        1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))
    )


def learning_rate(name: str, peak: float, *, epoch: int, step: int, total_steps: int,
                  batch: int = 1024, warmup_fraction: float | None = None,
                  min_lr: float | None = None) -> float:
    if name in {"glm5_cosine", "kimi3_cosine", "ds41_plateau_cosine"}:
        return recipe_learning_rate(name, peak, batch=batch, step=step, total_steps=total_steps,
                                    warmup_fraction=warmup_fraction, min_lr=min_lr)
    if name == "dinov2_step_cosine":
        return dinov2_learning_rate(peak, batch=batch, step=step, total_steps=total_steps,
                                   warmup_fraction=.16 if warmup_fraction is None else warmup_fraction,
                                   min_lr=1e-6 if min_lr is None else min_lr)
    if name == "sclong_epoch_restarts":
        return sclong_learning_rate(epoch, max_lr=peak)
    if name != "legacy_step_cosine":
        raise ValueError(f"Unknown learning-rate schedule: {name}")
    warmup = max(1, math.ceil(total_steps * 0.1))
    factor = (
        (step + 1) / warmup
        if step < warmup
        else 0.01 + 0.99 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup - 1))) / 2
    )
    return peak * factor
