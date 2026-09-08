"""Native teacher/student five-loss orchestration for continuous cell crops."""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import BackboneConfig, CellBackbone
from .distillation import (
    ProjectionHead,
    TeacherCenter,
    dino_loss,
    koleo_loss,
    pretraining_loss,
    reconstruction_loss,
    update_teacher,
)


@dataclass(frozen=True)
class HeadConfig:
    hidden: int = 2048
    bottleneck: int = 256
    cell_prototypes: int = 8192
    gene_prototypes: int = 4096
    ibot_separate_head: bool = False


class PretrainingNetwork(nn.Module):
    def __init__(self, backbone_config: BackboneConfig, head_config: HeadConfig):
        super().__init__()
        self.backbone = CellBackbone(backbone_config)
        width = backbone_config.width
        self.cell_head = ProjectionHead(width, head_config.cell_prototypes, head_config.hidden, head_config.bottleneck)
        self.gene_head = (
            ProjectionHead(width, head_config.gene_prototypes, head_config.hidden, head_config.bottleneck)
            if head_config.ibot_separate_head else self.cell_head
        )
        self.gene_expression = nn.Sequential(
            nn.Linear(width, width, bias=False), nn.LeakyReLU(0.2), nn.Linear(width, 1, bias=False)
        )
        self.cell_gene_query = nn.Linear(width, width, bias=False)

    def encode(self, view, *, clean=False):
        return self.backbone(
            view["gene_ids"],
            view["expression"],
            view["valid"],
            None if clean else view["hidden"],
        )

    def reconstruct(self, encoded):
        token = self.gene_expression(encoded["genes"]).squeeze(-1)
        query = self.cell_gene_query(encoded["gene_identity"]).sigmoid()
        cell = (query * encoded["cls"].unsqueeze(1)).sum(-1)
        return token, cell


class PretrainingSystem(nn.Module):
    """Wrap once in DDP; centers use explicit global sums/counts.

    Only selected hidden token features enter the large prototype head. This is
    equivalent to masking the full token CE but avoids allocating B*T*4096 logits.
    Call update_ema exactly once after a successful optimizer step, never per
    gradient-accumulation microbatch. Centers update per training forward.
    """

    def __init__(self, config: BackboneConfig, heads: HeadConfig | None = None):
        super().__init__()
        heads = HeadConfig() if heads is None else heads
        self.student = PretrainingNetwork(config, heads)
        self.teacher = deepcopy(self.student).requires_grad_(False).eval()
        self.cell_center = TeacherCenter(heads.cell_prototypes)
        self.gene_center = TeacherCenter(heads.gene_prototypes if heads.ibot_separate_head else heads.cell_prototypes)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def update_ema(self, completed_steps: int, total_steps: int):
        if not 1 <= completed_steps <= total_steps:
            raise ValueError("Invalid completed optimizer step count")
        # DINOv2 indexes its schedule at the zero-based optimizer iteration,
        # then applies EMA after optimizer.step(). Last in-budget value is
        # close to, not exactly, 1 (official cosine denominator is total_steps).
        progress = (completed_steps - 1) / total_steps
        momentum = 1 - (1 - 0.994) * (math.cos(math.pi * progress) + 1) / 2
        update_teacher(self.student, self.teacher, momentum)
        return momentum

    def forward(self, batch, *, step: int, total_steps: int):
        views = batch["views"]
        if len(views) < 2 or len(batch["cell_ids"]) < 2:
            raise ValueError("Pretraining needs two globals and >=2 cells")
        if len(set(batch["cell_ids"])) != len(batch["cell_ids"]):
            raise ValueError("KoLeo batch repeats a cell")
        if total_steps < 1 or not 0 <= step < total_steps:
            raise ValueError("Invalid optimizer step")
        temperature = 0.04 + 0.03 * min(1, step / max(1, math.ceil(total_steps * 0.1)))
        t_cell_logits, t_gene_logits, t_prob = [], [], []
        with torch.no_grad():
            for view in views[:2]:
                encoded = self.teacher.encode(view, clean=True)
                logits = self.teacher.cell_head(encoded["cls"])
                t_cell_logits.append(logits)
                hidden = view["hidden"] & view["valid"]
                token_logits = self.teacher.gene_head(encoded["genes"][hidden])
                t_gene_logits.append(token_logits)
                t_prob.append(self.gene_center.targets(token_logits, temperature))
        teacher_cells = [self.cell_center.targets(logits, temperature) for logits in t_cell_logits]
        student_cells, expressions, cell_expressions, ibots, koleos = [], [], [], [], []
        for index, view in enumerate(views):
            encoded = self.student.encode(view)
            student_cells.append(self.student.cell_head(encoded["cls"]))
            if index >= 2:
                continue
            prediction, cell_prediction = self.student.reconstruct(encoded)
            targets = view["targets"] & view["valid"]
            expressions.append(reconstruction_loss(prediction, view["expression"], targets))
            cell_expressions.append(reconstruction_loss(cell_prediction, view["expression"], targets))
            hidden = view["hidden"] & view["valid"]
            logits = self.student.gene_head(encoded["genes"][hidden])
            ce = -(t_prob[index] * F.log_softmax(logits.float() / 0.1, -1)).sum(-1)
            weights = (hidden.sum(-1).clamp_min(1).reciprocal().unsqueeze(-1).expand_as(hidden))[hidden]
            ibots.append((ce * weights).sum() / hidden.shape[0])
            koleos.append(koleo_loss(encoded["cls"]))
        losses = {
            "expression": torch.stack(expressions).mean(),
            "cell_expression": torch.stack(cell_expressions).mean(),
            "dino": dino_loss(student_cells, teacher_cells),
            "ibot": torch.stack(ibots).mean(),
            "koleo": torch.stack(koleos).sum(),
        }
        losses["total"] = pretraining_loss(**losses, step=step, total_steps=total_steps)
        if self.training:
            self.cell_center.update(torch.cat(t_cell_logits))
            self.gene_center.update(torch.cat(t_gene_logits))
        return losses
