"""Native LoRA linear adapters. Source comparison is recorded in the source ledger."""

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Frozen base plus alpha/r * B A; no mutation on train/eval transitions."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 32, dropout: float = 0.05):
        super().__init__()
        if rank < 1 or rank > min(base.in_features, base.out_features):
            raise ValueError("LoRA rank must be positive and no larger than either feature dimension")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA scaling/dropout")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.dropout(x), self.a), self.b) * self.scale


def attach_lora(
    backbone: nn.Module, target_paths: Sequence[str], *, rank: int = 16, alpha: float = 32, dropout: float = 0.05
) -> list[str]:
    """Freeze backbone and adapt an explicit, audited list of exact module paths.

    Validate the complete selection before mutation; typos/duplicates/overlapping
    existing adapters are errors. New task heads should live outside backbone.
    """
    if not target_paths or len(set(target_paths)) != len(target_paths):
        raise ValueError("LoRA requires nonempty, unique target paths")
    replacements = []
    for path in target_paths:
        if not path or any(not part for part in path.split(".")):
            raise ValueError("Invalid module path")
        module = backbone.get_submodule(path)
        if not isinstance(module, nn.Linear) or any(isinstance(x, LoRALinear) for x in backbone.modules()):
            raise ValueError("Targets must be ordinary Linear modules on an unadapted backbone")
        if rank < 1 or rank > min(module.in_features, module.out_features):
            raise ValueError(f"Rank does not fit {path}")
        if not math.isfinite(alpha) or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA scaling/dropout")
        parent, _, leaf = path.rpartition(".")
        replacements.append((backbone.get_submodule(parent) if parent else backbone, leaf, module))
    backbone.requires_grad_(False)
    for parent, leaf, module in replacements:
        setattr(parent, leaf, LoRALinear(module, rank, alpha, dropout))
    return list(target_paths)
