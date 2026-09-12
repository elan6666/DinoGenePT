"""Native population-matched knowledge-local perturbation model.

This is our declared stage-2 adaptation, not a reproduction of unavailable
DINOcell training code. Frozen external vectors enter only via source adapters.
Observed/teacher post-treatment cells are training-only inputs; predict() has no
post-treatment argument. Control and post bags are not paired single cells.
"""

import math
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .distillation import TeacherCenter, update_teacher
from .lora import attach_lora
from .pretraining import PretrainingNetwork
from .schedule import teacher_momentum

SOURCES = ("TextBase", "GO", "Protein", "Pathway", "HPA")


@dataclass(frozen=True)
class PerturbationConfig:
    main_anchor: str = "TextBase"
    vector_width: int = 2048
    decoder_width: int = 128
    lora_rank: int = 16
    lora_alpha: float = 32
    lora_dropout: float = 0.05
    teacher_temperature: float = 0.07
    student_temperature: float = 0.1
    distillation_weight: float = 0.1
    local_sources: tuple[str, ...] = SOURCES

    def __post_init__(self):
        if self.main_anchor not in {"CellGene", "TextBase"}:
            raise ValueError("Main anchor must be CellGene or TextBase")
        if min(self.vector_width, self.decoder_width, self.teacher_temperature, self.student_temperature) <= 0:
            raise ValueError("Invalid perturbation dimensions/temperatures")
        if self.distillation_weight != 0.1:
            raise ValueError("This campaign uses the fixed 0.1 distillation weight, not a loss ablation")
        if len(set(self.local_sources)) != len(self.local_sources) or set(self.local_sources) - set(SOURCES):
            raise ValueError("Invalid knowledge source allowlist")


class ConditionalNetwork(nn.Module):
    def __init__(self, pretrained: PretrainingNetwork, config: PerturbationConfig):
        super().__init__()
        self.backbone = deepcopy(pretrained.backbone)
        self.cell_head = deepcopy(pretrained.cell_head)
        self.config = config
        attach_lora(
            self.backbone,
            self.backbone.lora_targets(),
            rank=config.lora_rank,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        width = self.backbone.config.width
        self.adapters = nn.ModuleDict(
            {
                source: nn.Sequential(nn.Linear(config.vector_width, width, bias=False), nn.LayerNorm(width))
                for source in SOURCES
            }
        )
        # Factorized CLS/gene decoder avoids a B*G*768 expression-token tensor.
        self.cell_query = nn.Linear(width, config.decoder_width, bias=False)
        self.gene_query = nn.Linear(width, config.decoder_width, bias=False)
        self.control_query = nn.Linear(1, config.decoder_width, bias=False)
        self.delta_scale = nn.Parameter(torch.ones(()))

    def _anchor(self, targets, source, vectors):
        if targets.ndim != 1 or targets.numel() < 1 or targets.unique().numel() != targets.numel():
            raise ValueError("One condition requires distinct, nonempty perturbation gene IDs")
        if targets.dtype != torch.long or (targets < 1).any() or (targets > self.backbone.config.genes).any():
            raise ValueError("Target gene missing from frozen vocabulary")
        order = targets.argsort()
        if source == "CellGene":
            return self.backbone.gene(targets[order]).detach()
        if source not in SOURCES or vectors is None:
            raise ValueError(f"Missing required anchor {source}")
        if vectors.shape != (len(targets), self.config.vector_width) or not torch.isfinite(vectors).all():
            raise ValueError("Source requires a finite vector for EVERY perturbation target")
        if (vectors.float().norm(dim=-1) == 0).any():
            raise ValueError("Zero-filled missing knowledge vectors are forbidden")
        return self.adapters[source](F.normalize(vectors.detach().float()[order], dim=-1))

    def encode(self, view, targets=None, source=None, vectors=None):
        prefix = (
            None
            if source is None
            else self._anchor(targets, source, vectors).unsqueeze(0).expand(view["gene_ids"].shape[0], -1, -1)
        )
        return self.backbone(
            view["gene_ids"], view["expression"], view["valid"], view.get("hidden"), prefix_tokens=prefix
        )

    def decode(self, cls, control, axis):
        if axis.ndim != 1 or axis.dtype != torch.long or axis.unique().numel() != axis.numel():
            raise ValueError("Prediction axis must be unique gene IDs")
        if (axis < 1).any() or (axis > self.backbone.config.genes).any():
            raise ValueError("Prediction axis missing from frozen vocabulary")
        if control.shape != (len(cls), len(axis)) or not torch.isfinite(control).all() or (control < 0).any():
            raise ValueError("Control must be finite nonnegative continuous expression on the full axis")
        gene = self.gene_query(self.backbone.gene(axis)).unsqueeze(0)
        query = (gene + self.control_query(control.unsqueeze(-1))).sigmoid()
        delta = (query * self.cell_query(cls).unsqueeze(1)).sum(-1) / math.sqrt(self.config.decoder_width)
        return control + self.delta_scale * delta


class PerturbationSystem(nn.Module):
    def __init__(self, pretrained: PretrainingNetwork, config: PerturbationConfig | None = None):
        super().__init__()
        self.config = PerturbationConfig() if config is None else config
        self.student = ConditionalNetwork(pretrained, self.config)
        self.teacher = deepcopy(self.student).requires_grad_(False).eval()
        prototypes = pretrained.cell_head.prototypes.out_features
        self.center = TeacherCenter(prototypes)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def predict(self, control_view, full_control, axis, targets, main_vectors=None):
        """Inference uses only controls and the declared main anchor, no locals/post."""
        encoded = self.student.encode(control_view, targets, self.config.main_anchor, main_vectors)
        return self.student.decode(encoded["cls"], full_control, axis)

    def forward(
        self, *, control_view, observed_view, teacher_view, full_control, train_post, axis, targets, source_vectors
    ):
        if not self.training:
            raise RuntimeError("Post-treatment teacher views are training-only; use predict for evaluation")
        if set(source_vectors) - set(SOURCES):
            raise ValueError("Unknown knowledge source")
        if train_post.ndim != 2 or train_post.shape[1] != len(axis) or len(train_post) < 1:
            raise ValueError("Post-treatment bag must share the full prediction gene axis")
        if not torch.isfinite(train_post).all() or (train_post < 0).any():
            raise ValueError("Post-treatment training targets must be finite and nonnegative")
        with torch.no_grad():
            teacher_cls = self.teacher.encode(teacher_view)["cls"]
            teacher_logits = self.teacher.cell_head(teacher_cls)
            # Mean of per-cell probabilities, NOT softmax of mean logits.
            target = self.center.targets(teacher_logits, self.config.teacher_temperature).mean(0, keepdim=True)

        def distill(cls):
            logits = self.student.cell_head(cls).float() / self.config.student_temperature
            return -(target * logits.log_softmax(-1)).sum(-1).mean()

        main = self.config.main_anchor
        encoded = self.student.encode(control_view, targets, main, source_vectors.get(main))
        prediction = self.student.decode(encoded["cls"], full_control, axis)
        reconstruction = F.mse_loss(prediction.float().mean(0), train_post.detach().float().mean(0))
        primary = distill(encoded["cls"])
        local_losses = []
        active_sources = []
        for source in self.config.local_sources:
            vectors = source_vectors.get(source)
            if source == main or vectors is None:
                continue
            local_losses.append(distill(self.student.encode(control_view, targets, source, vectors)["cls"]))
            active_sources.append(source)
        # When TextBase is primary, pretrained gene tokens are not silently added
        # as an extra local; that belongs to the separately named anchor ablation.
        knowledge = torch.stack(local_losses).mean() if local_losses else primary * 0
        observed = distill(self.student.encode(observed_view)["cls"])
        total = reconstruction + self.config.distillation_weight * (primary + knowledge + observed)
        self.center.update(teacher_logits)
        return {
            "total": total,
            "reconstruction": reconstruction,
            "primary_dino": primary,
            "knowledge_dino": knowledge,
            "observed_dino": observed,
            "local_count": len(active_sources),
        }

    @torch.no_grad()
    def update_ema(self, completed_steps, total_steps):
        momentum = teacher_momentum(completed_steps, total_steps)
        update_teacher(self.student, self.teacher, momentum)
        return momentum
