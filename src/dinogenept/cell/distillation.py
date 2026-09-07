"""DINO-family primitives, independent from upstream runtime packages.

Project adaptations: same-global exclusion and pair averaging, explicit
gene-aligned iBOT masks, per-cell reduction, strict self-exclusion in KoLeo.
"""

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


class ProjectionHead(nn.Module):
    def __init__(self, width: int, prototypes: int, hidden: int = 2048, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, bottleneck)
        )
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)
        self.prototypes = weight_norm(nn.Linear(bottleneck, prototypes, bias=False))
        with torch.no_grad():
            self.prototypes.parametrizations.weight.original0.fill_(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        return self.prototypes(F.normalize(x, dim=-1, eps=1e-6 if x.dtype == torch.float16 else 1e-12))


class TeacherCenter(nn.Module):
    """Explicit post-target update, using DDP sums/counts, not mean-of-rank-means."""

    def __init__(self, prototypes: int, momentum: float = 0.9):
        super().__init__()
        if not 0 <= momentum < 1:
            raise ValueError("Invalid center momentum")
        self.momentum = momentum
        self.register_buffer("center", torch.zeros(prototypes))

    @torch.no_grad()
    def targets(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Invalid teacher temperature")
        return F.softmax((logits.float() - self.center.float()) / temperature, dim=-1)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, valid: torch.Tensor | None = None) -> None:
        if logits.shape[-1] != self.center.numel():
            raise ValueError("Prototype axis mismatch")
        rows = logits.detach().float().reshape(-1, logits.shape[-1])
        if valid is not None:
            if valid.shape != logits.shape[:-1] or valid.dtype != torch.bool:
                raise ValueError("Center validity mask mismatch")
            rows = rows[valid.reshape(-1)]
        total = rows.sum(0)
        count = total.new_tensor(float(rows.shape[0]))
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(total)
            dist.all_reduce(count)
        if count.item() > 0:
            self.center.lerp_((total / count).to(self.center.dtype), 1 - self.momentum)


def dino_loss(student: Sequence[torch.Tensor], teacher: Sequence[torch.Tensor], temperature: float = 0.1):
    """Student order: matching global0/global1, then locals; teacher has two globals."""
    if len(teacher) != 2 or len(student) < 2 or temperature <= 0:
        raise ValueError("DINO requires two globals and a positive temperature")
    terms = []
    for ti, target in enumerate(teacher):
        for si, logits in enumerate(student):
            if si == ti:
                continue
            if logits.shape != target.shape or logits.ndim != 2:
                raise ValueError("DINO views must have identical cell/prototype axes")
            terms.append(-(target.detach().float() * F.log_softmax(logits.float() / temperature, -1)).sum(-1))
    return torch.stack(terms).mean()


def masked_cell_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or mask.shape != values.shape or mask.dtype != torch.bool:
        raise ValueError("Expected values and boolean mask [cell,gene]")
    # where, not multiplication: non-selected NaNs must not pollute the loss.
    sums = torch.where(mask, values, 0).sum(-1)
    counts = mask.sum(-1).clamp_min(1)
    return (sums / counts).mean()


def ibot_loss(student: torch.Tensor, teacher: torch.Tensor, hidden: torch.Tensor, temperature: float = 0.1):
    """Inputs already aligned to the same global's gene IDs; hidden excludes pad."""
    if student.shape != teacher.shape or student.ndim != 3 or temperature <= 0:
        raise ValueError("iBOT requires aligned [cell,gene,prototype] tensors")
    per_gene = -(teacher.detach().float() * F.log_softmax(student.float() / temperature, -1)).sum(-1)
    return masked_cell_mean(per_gene, hidden)


def reconstruction_loss(prediction: torch.Tensor, truth: torch.Tensor, targets: torch.Tensor):
    if prediction.shape != truth.shape:
        raise ValueError("Reconstruction axes differ")
    return masked_cell_mean((prediction.float() - truth.float()).square(), targets)


def koleo_loss(cls: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """One global at a time, per-rank distinct-cell batch; call twice and sum."""
    if cls.ndim != 2 or cls.shape[0] < 2:
        raise ValueError("KoLeo needs >=2 distinct cells per forward microbatch")
    with torch.autocast(device_type=cls.device.type, enabled=False):
        normalized = F.normalize(cls.float(), dim=-1, eps=eps)
        similarity = normalized @ normalized.T
        similarity.fill_diagonal_(-torch.inf)
        neighbors = similarity.detach().argmax(-1)
        # Match DINOv2's PairwiseDistance epsilon convention, including at ties.
        distance = F.pairwise_distance(normalized, normalized[neighbors], p=2, eps=eps)
        return -(distance + eps).log().mean()


@torch.no_grad()
def update_teacher(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    """EMA all corresponding parameters; exact buffer copy, no silent zip truncation."""
    if not 0 <= momentum <= 1:
        raise ValueError("Invalid EMA momentum")
    sp, tp = dict(student.named_parameters()), dict(teacher.named_parameters())
    sb, tb = dict(student.named_buffers()), dict(teacher.named_buffers())
    if sp.keys() != tp.keys() or sb.keys() != tb.keys():
        raise ValueError("Student/teacher structures differ")
    if any(sp[key].shape != value.shape for key, value in tp.items()):
        raise ValueError("Student/teacher parameter shape mismatch")
    if any(sb[key].shape != value.shape for key, value in tb.items()):
        raise ValueError("Student/teacher buffer shape mismatch")
    for key, value in tp.items():
        value.lerp_(sp[key].detach().to(value.dtype), 1 - momentum)
        value.requires_grad_(False)
    for key, value in tb.items():
        value.copy_(sb[key])
    teacher.eval()


def pretraining_loss(expression, cell_expression, dino, ibot, koleo, *, step: int, total_steps: int):
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError("Invalid pretraining step budget")
    ramp = min(1.0, step / max(1, math.ceil(total_steps * 0.1)))
    return expression + cell_expression + ramp * (dino + ibot + 0.1 * koleo)
