"""Losses kept separate so every term can be ablated independently."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def condition_dino_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    center: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
    reduction: str = "mean",
) -> torch.Tensor:
    student_log_prob = F.log_softmax(student_logits / student_temperature, dim=-1)
    teacher_prob = F.softmax((teacher_logits.detach() - center) / teacher_temperature, dim=-1)
    losses = -(teacher_prob * student_log_prob).sum(dim=-1)
    if reduction == "none":
        return losses
    if reduction != "mean":
        raise ValueError(f"unsupported DINO reduction: {reduction!r}")
    return losses.mean()


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_logits: torch.Tensor, momentum: float) -> None:
    batch_center = teacher_logits.mean(dim=0, keepdim=True)
    center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)


def prediction_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    control: torch.Tensor,
    *,
    kind: str,
    direction_weight: float,
) -> torch.Tensor:
    mse = F.mse_loss(prediction, truth)
    if kind == "mse":
        return mse
    if kind != "direction_aware":
        raise ValueError(f"unknown prediction loss: {kind!r}")
    true_delta = truth - control
    pred_delta = prediction - control
    direction = F.relu(-(true_delta * pred_delta)).mean()
    return mse + direction_weight * direction


def delta_ibot_loss(
    token_hidden: torch.Tensor,
    gene_indices: torch.Tensor,
    expression_mask: torch.Tensor,
    delta_target: torch.Tensor,
    token_head,
) -> torch.Tensor:
    prediction = token_head(token_hidden).squeeze(-1)
    targets = delta_target.gather(1, gene_indices)
    selected = expression_mask.bool()
    if not selected.any():
        return prediction.sum() * 0.0
    return F.mse_loss(prediction[selected], targets[selected])


def koleo_loss(features: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    if features.shape[0] < 2:
        return features.sum() * 0.0
    normalized = F.normalize(features, dim=-1)
    distances = torch.cdist(normalized, normalized, p=2)
    diagonal = torch.eye(
        distances.shape[0], dtype=torch.bool, device=distances.device
    )
    distances = distances.masked_fill(diagonal, float("inf"))
    nearest = distances.min(dim=1).values
    return -torch.log(nearest.clamp_min(epsilon)).mean()
