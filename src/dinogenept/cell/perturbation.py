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
from .ibot_chunk import projected_ibot
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
    reconstruction_weight: float = 0.5
    primary_weight: float = 1.0
    knowledge_weight: float = 1.0
    observed_weight: float = 1.0
    observed_ibot: bool = False
    observed_koleo: bool = False
    ibot_weight: float = 0.5
    koleo_weight: float = 0.1
    ibot_chunk_size: int = 0
    local_sources: tuple[str, ...] = SOURCES

    def __post_init__(self):
        if self.main_anchor not in {"CellGene", "TextBase"}:
            raise ValueError("Main anchor must be CellGene or TextBase")
        if min(self.vector_width, self.decoder_width, self.teacher_temperature, self.student_temperature) <= 0:
            raise ValueError("Invalid perturbation dimensions/temperatures")
        weights = (self.reconstruction_weight, self.primary_weight, self.knowledge_weight,
                   self.observed_weight, self.ibot_weight, self.koleo_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights) or self.ibot_chunk_size < 0:
            raise ValueError("Invalid loss weights/chunk size")
        if len(set(self.local_sources)) != len(self.local_sources) or set(self.local_sources) - set(SOURCES):
            raise ValueError("Invalid knowledge source allowlist")

    def loss_weights(self):
        return dict(reconstruction=self.reconstruction_weight, primary_dino=self.primary_weight,
                    knowledge_dino=self.knowledge_weight, observed_dino=self.observed_weight,
                    observed_ibot=self.ibot_weight if self.observed_ibot else 0.0,
                    observed_koleo=self.koleo_weight if self.observed_koleo else 0.0)


def condition_filtered_koleo(cls, conditions, eps=1e-8):
    """Project adaptation: exclude same-condition candidates, no history/all-gather."""
    if conditions is None or conditions.shape != (len(cls),):
        raise ValueError("KoLeo requires one condition ID per observed cell")
    with torch.autocast(device_type=cls.device.type, enabled=False):
        u = F.normalize(cls.float(), dim=-1, eps=eps)
        candidates = conditions[:, None] != conditions[None, :]
        eligible = candidates.any(-1)
        if not eligible.any():
            return cls.sum() * 0, 0
        scores = (u @ u.T).detach().masked_fill(~candidates, -torch.inf)
        nearest = scores.argmax(-1)
        distance = F.pairwise_distance(u[eligible], u[nearest[eligible]], eps=eps)
        return -(distance + eps).log().mean(), int(eligible.sum())


class ConditionalNetwork(nn.Module):
    def __init__(self, pretrained: PretrainingNetwork, config: PerturbationConfig):
        super().__init__()
        self.backbone = deepcopy(pretrained.backbone)
        self.cell_head = deepcopy(pretrained.cell_head)
        # No extra parameters or state when iBOT is disabled.
        if config.observed_ibot:
            self.gene_head = (self.cell_head if pretrained.gene_head is pretrained.cell_head
                              else deepcopy(pretrained.gene_head))
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
        if self.config.observed_ibot:
            self.gene_center = TeacherCenter(self.student.gene_head.prototypes.out_features)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def predict(self, control_view, full_control, axis, targets, main_vectors=None):
        """Inference uses only controls and the declared main anchor, no locals/post."""
        encoded = self.student.encode(control_view, targets, self.config.main_anchor, main_vectors)
        return self.student.decode(encoded["cls"], full_control, axis)

    def forward(
        self, *, control_view, observed_view, teacher_view, full_control, train_post, axis, targets, source_vectors,
        observed_conditions=None, observed_cell_ids=None, teacher_cell_ids=None
    ):
        if not self.training:
            raise RuntimeError("Post-treatment teacher views are training-only; use predict for evaluation")
        if set(source_vectors) - set(SOURCES):
            raise ValueError("Unknown knowledge source")
        if observed_view is not None:
            if (observed_cell_ids is None or teacher_cell_ids is None
                    or observed_cell_ids.shape != (len(observed_view["gene_ids"]),)
                    or teacher_cell_ids.shape != (len(teacher_view["gene_ids"]),)):
                raise ValueError("Observed/teacher cell IDs are required and must match views")
            if (observed_cell_ids.unique().numel() != observed_cell_ids.numel()
                    or teacher_cell_ids.unique().numel() != teacher_cell_ids.numel()
                    or torch.isin(observed_cell_ids, teacher_cell_ids).any()):
                raise ValueError("Observed/teacher cell IDs must be unique and disjoint")
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
        observed = ibot = koleo = primary * 0
        masked_cells = eligible_cells = 0
        observed_count = 0 if observed_view is None else len(observed_view["gene_ids"])
        if observed_count:
            obs = self.student.encode(observed_view)
            observed = distill(obs["cls"])
            if self.config.observed_ibot:
                hidden = observed_view["hidden"] & observed_view["valid"]
                counts = hidden.sum(-1)
                masked_cells = int((counts > 0).sum())
                if masked_cells:
                    # Same B, identical gene order and values, ONLY hidden mask removed.
                    clean = {**observed_view, "hidden": torch.zeros_like(hidden)}
                    with torch.no_grad():
                        clean_encoded = self.teacher.encode(clean)
                    weights = (1 / counts.clamp_min(1).float() / masked_cells)[:, None].expand_as(hidden)[hidden]
                    ibot, total_logits, count = projected_ibot(
                        self.student.gene_head, self.teacher.gene_head, obs["genes"][hidden],
                        clean_encoded["genes"][hidden], weights, self.gene_center.center,
                        self.config.teacher_temperature, self.config.student_temperature, self.config.ibot_chunk_size,
                    )
                    self.gene_center.update_statistics(total_logits, count)
            if self.config.observed_koleo:
                koleo, eligible_cells = condition_filtered_koleo(obs["cls"], observed_conditions)
        losses = dict(reconstruction=reconstruction, primary_dino=primary, knowledge_dino=knowledge,
                      observed_dino=observed, observed_ibot=ibot, observed_koleo=koleo)
        total = sum(losses[name] * weight for name, weight in self.config.loss_weights().items())
        self.center.update(teacher_logits)
        return {
            "total": total,
            **losses,
            "observed_cells": observed_count,
            "ibot_masked_cells": masked_cells,
            "ibot_participation": masked_cells / max(1, observed_count),
            "koleo_eligible_cells": eligible_cells,
            "koleo_eligible_fraction": eligible_cells / max(1, observed_count),
            "local_count": len(active_sources),
        }

    @torch.no_grad()
    def update_ema(self, completed_steps, total_steps):
        momentum = teacher_momentum(completed_steps, total_steps)
        update_teacher(self.student, self.teacher, momentum)
        return momentum
