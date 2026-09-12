"""Native dense Muon adapters with PyTorch AdamW auxiliary groups.

NS polynomial and Nesterov convention audited against KellerJordan/Muon
f98f1cacc0263b04290753e32be8d498c1efc806. FP32 NS is our stability
adaptation; no extra optimizer collectives (DDP already synchronizes grads).
Head layouts are model-aware, including interleaved MLA K/V rows.
"""

import math

import torch
from torch import nn

from .backbone import ERetMixer, GatedGQA, GatedMLA, KDAMixer


def orthogonalize(matrix, steps=5):
    if matrix.ndim < 2 or steps < 1:
        raise ValueError("Muon requires a matrix and positive NS steps")
    x = matrix.float()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.mT
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = x @ x.mT
        x = 3.4445 * x + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ x
    return x.mT if transposed else x


def sinkhorn_update(update, steps=11, threshold=1e-3, eps=1e-20):
    """DS4.1 Algorithm 1: odd alternating L2 normalizations, final row RMS.

    FP32 only; near-zero mask uses Nesterov row norm relative to its mean.
    No second-moment state and no weight decay. Not a row-freezing optimizer.
    """
    if update.ndim != 2 or steps < 1 or steps % 2 != 1 or threshold < 0 or eps <= 0:
        raise ValueError("Invalid Sinkhorn update")
    u = update.float().clone()
    norms = u.norm(dim=1, keepdim=True)
    u.masked_fill_(norms <= threshold * norms.mean(), 0)
    for k in range(steps):
        u = u / (u.norm(dim=1 if k % 2 == 0 else 0, keepdim=True) + eps)
    return u * math.sqrt(u.shape[1])


def head_layout(mixer, leaf, mode):
    """Complete partition of rows; None means one whole matrix."""
    if mode == "muon":
        return None
    include_value = mode in {"glm5_muon", "kimi3_muon"}
    if isinstance(mixer, (KDAMixer, ERetMixer, GatedGQA)):
        if leaf not in ({"q_proj", "k_proj", "v_proj"} if include_value else {"q_proj", "k_proj"}):
            return None
        heads = mixer.kv_heads if isinstance(mixer, GatedGQA) and leaf in {"k_proj", "v_proj"} else mixer.heads
        return [list(range(h * mixer.dim, (h + 1) * mixer.dim)) for h in range(heads)]
    if isinstance(mixer, GatedMLA):
        if leaf == "q_b_proj":
            size = mixer.dim + mixer.shared
            return [list(range(h * size, (h + 1) * size)) for h in range(mixer.heads)]
        if leaf == "kv_b_proj":
            d = mixer.dim
            keys = [list(range(h * 2 * d, h * 2 * d + d)) for h in range(mixer.heads)]
            values = [list(range(h * 2 * d + d, (h + 1) * 2 * d)) for h in range(mixer.heads)]
            return keys + (values if include_value else [sum(values, [])])
    return None


class HybridMuon(torch.optim.AdamW):
    """A single checkpointable optimizer with native Muon and auxiliary AdamW.

    Absent gradients skip updates, as PyTorch AdamW does. A zero gradient with
    historical momentum is NOT an absent gradient and can update a parameter.
    All gradients must be finite before entry; runner clips/checks globally.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        custom = [(g, p, p.grad) for g in self.param_groups if g.get("algorithm") in {"muon", "sinkhorn"}
                  for p in g["params"] if p.grad is not None]
        for _, p, _ in custom:
            p.grad = None
        try:
            super().step()
        finally:
            for _, p, grad in custom:
                p.grad = grad
        for group, p, grad in custom:
            if grad.is_sparse:
                raise ValueError("Sparse Muon gradients are not supported")
            state = self.state[p]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(p, dtype=torch.float32)
                state["step"] = torch.zeros(())
            state["step"] += 1
            momentum = state["momentum_buffer"]
            momentum.lerp_(grad.float(), 1 - group["momentum"])
            update = grad.float().lerp(momentum, group["momentum"])
            if group["algorithm"] == "sinkhorn":
                delta = sinkhorn_update(update)
                p.add_(delta.to(p.dtype), alpha=-group["lr"] * group["rms_scale"])
                continue
            p.mul_(1 - group["lr"] * group["weight_decay"])
            layout = group["layout"]
            if layout is None:
                delta = orthogonalize(update, group["ns_steps"])
                scale = group["rms_scale"] * math.sqrt(max(update.shape))
                p.add_(delta.to(p.dtype), alpha=-group["lr"] * scale)
            else:
                # Equal-shaped heads share a batched NS launch, but never a
                # normalization norm. DS's combined V remains its own bucket.
                buckets = {}
                for rows in layout:
                    buckets.setdefault(len(rows), []).append(rows)
                for rows in buckets.values():
                    index = torch.tensor(rows, device=p.device)
                    delta = orthogonalize(update[index], group["ns_steps"])
                    scale = group["rms_scale"] * math.sqrt(max(delta.shape[-2:]))
                    p.index_add_(0, index.flatten(), delta.flatten(0, 1).to(p.dtype), alpha=-group["lr"] * scale)
        return loss


def build_optimizer(student, training):
    from .optimization import optimizer_groups

    mode = training.get("optimizer", "adamw")
    if mode == "adamw":
        if training.get("embedding_optimizer", "adamw") != "adamw" or training.get("qk_clip_threshold") is not None:
            raise ValueError("AdamW cannot silently enable Sinkhorn or MuonClip")
        return torch.optim.AdamW(optimizer_groups(student, training["weight_decay"]),
                                 lr=training["learning_rate"], betas=tuple(training["betas"]))
    if mode not in {"muon", "glm5_muon", "kimi3_muon", "ds41_muon"}:
        raise ValueError(f"Unknown optimizer: {mode}")
    beta = training.get("muon_momentum", .95)
    ns = training.get("muon_ns_steps", 5)
    scale = training.get("muon_rms_scale", .18 if mode == "ds41_muon" else .2)
    multiplier = training.get("muon_lr_multiplier", 1.)
    if (not 0 <= beta < 1 or type(ns) is not int or ns < 1 or not math.isfinite(scale) or scale <= 0
            or not math.isfinite(multiplier) or multiplier <= 0):
        raise ValueError("Invalid Muon hyperparameters")
    groups, assigned = [], set()
    if training.get("embedding_optimizer", "adamw") == "sinkhorn":
        if mode != "ds41_muon":
            raise ValueError("Sinkhorn embedding requires the DS isolated extension")
        p = student.backbone.gene.weight
        if not p.requires_grad:
            raise ValueError("Cannot compare optimizer on a frozen embedding")
        groups.append(dict(params=[p], algorithm="sinkhorn", momentum=beta, rms_scale=.18,
                           lr_scale=1., weight_decay=0.))
        assigned.add(id(p))
    elif training.get("embedding_optimizer", "adamw") != "adamw":
        raise ValueError("Unknown embedding optimizer")
    for i, block in enumerate(student.backbone.blocks):
        for branch_name in ("mixer", "ffn"):
            branch = getattr(block, branch_name)
            for name, module in branch.named_modules():
                if not isinstance(module, nn.Linear) or not module.weight.requires_grad:
                    continue
                p = module.weight
                if p.dtype != torch.float32:
                    raise ValueError("Native Muon requires FP32 master parameters; use BF16 autocast for activations")
                layout = head_layout(branch, name, mode) if branch_name == "mixer" else None
                groups.append(dict(params=[p], algorithm="muon", layout=layout, momentum=beta, ns_steps=ns,
                                   rms_scale=scale, lr_scale=multiplier, weight_decay=training["weight_decay"],
                                   parameter_names=[f"backbone.blocks.{i}.{branch_name}.{name}.weight"]))
                assigned.add(id(p))
    embeddings = {id(m.weight) for m in student.modules() if isinstance(m, nn.Embedding)}
    for group in optimizer_groups(student, training["weight_decay"]):
        params = [p for p in group["params"] if id(p) not in assigned]
        if params:
            groups.append(dict(params=params, algorithm="adamw", lr_scale=1., weight_decay=group["weight_decay"]))
    actual = [id(p) for g in groups for p in g["params"]]
    expected = {id(p) for p in student.parameters() if p.requires_grad}
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer partition has duplicate/missing parameters")
    if any(g["weight_decay"] != 0 for g in groups for p in g["params"] if id(p) in embeddings):
        raise ValueError("Embedding weight decay is prohibited")
    optimizer = HybridMuon(groups, lr=training["learning_rate"], betas=tuple(training["betas"]))
    threshold = training.get("qk_clip_threshold")
    if threshold is not None:
        if mode != "kimi3_muon" or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("QK clip requires Kimi Muon and a positive threshold")
        if any(isinstance(m, GatedGQA) for m in student.backbone.modules()):
            raise ValueError("QK-Clip currently audited only for MLA, not grouped shared K")
        for module in student.backbone.modules():
            if isinstance(module, GatedMLA):
                module.qk_clip_threshold = threshold
    return optimizer


@torch.no_grad()
def apply_qk_clip(student):
    """K2 head-specific/private+shared-channel rule, after step and before EMA.

    Our shared channels have NoPE, but the same bilinear rescaling applies.
    Synchronize max across ranks AND all views/microbatches. Statistics are
    reset at optimizer boundaries and never copied into Teacher parameters.
    """
    triggered = 0
    for module in student.backbone.modules():
        if not isinstance(module, GatedMLA) or module.qk_clip_threshold is None:
            continue
        maximum = module.qk_max.clone()
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
        gamma = (module.qk_clip_threshold / maximum.clamp_min(1e-12)).clamp_max(1)
        q = module.q_b_proj.weight.view(module.heads, module.dim + module.shared, -1)
        kv = module.kv_b_proj.weight.view(module.heads, 2 * module.dim, -1)
        q[:, :module.dim].mul_(gamma.sqrt()[:, None, None])
        q[:, module.dim:].mul_(gamma[:, None, None])
        kv[:, :module.dim].mul_(gamma.sqrt()[:, None, None])
        triggered += int((gamma < 1).sum())
        module.qk_max.zero_()
    return triggered
