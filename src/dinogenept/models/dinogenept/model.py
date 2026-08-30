"""DinoGenePT model: shared cell Transformer with knowledge-conditioned views."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .modules import CellTransformer, DINOHead


@dataclass
class CellViewOutput:
    cls: torch.Tensor
    token_hidden: torch.Tensor
    gene_indices: torch.Tensor
    route_anchor: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class CellBackbone(nn.Module):
    def __init__(self, n_genes: int, config: dict[str, Any]) -> None:
        super().__init__()
        dimension = int(config["d_model"])
        self.n_genes = n_genes
        self.dimension = dimension
        self.max_seq_len = int(config["max_seq_len"])
        self.gene_embedding = nn.Embedding(n_genes, dimension)
        self.expression_encoder = nn.Sequential(
            nn.Linear(1, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.transformer = CellTransformer(config)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def select_genes(
        self, expression: torch.Tensor, gene_indices: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if expression.ndim != 2 or expression.shape[1] != self.n_genes:
            raise ValueError("expression matrix differs from the configured gene axis")
        count = min(self.max_seq_len, self.n_genes)
        if gene_indices is None:
            gene_indices = torch.topk(expression.abs(), count, dim=1, sorted=True).indices
        values = expression.gather(1, gene_indices)
        return gene_indices, values

    def embed_gene_tokens(
        self,
        gene_indices: torch.Tensor,
        values: torch.Tensor,
        expression_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        expression_tokens = self.expression_encoder(values.unsqueeze(-1))
        if expression_mask is not None:
            if expression_mask.shape != values.shape:
                raise ValueError("expression mask shape differs from selected gene values")
            expression_tokens = torch.where(
                expression_mask.unsqueeze(-1),
                self.mask_token.expand(values.shape[0], values.shape[1], -1),
                expression_tokens,
            )
        return self.gene_embedding(gene_indices) + expression_tokens

    def forward(
        self,
        expression: torch.Tensor,
        *,
        knowledge_tokens: torch.Tensor | None = None,
        knowledge_mask: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
        expression_mask: torch.Tensor | None = None,
        route_anchor: torch.Tensor | None = None,
    ) -> CellViewOutput:
        selected_indices, values = self.select_genes(expression, gene_indices)
        tokens = self.embed_gene_tokens(selected_indices, values, expression_mask)
        batch = expression.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        prefix = [cls]
        prefix_mask = [torch.zeros(batch, 1, dtype=torch.bool, device=expression.device)]
        if knowledge_tokens is not None:
            if knowledge_mask is None:
                raise ValueError("knowledge tokens require a token mask")
            prefix.append(knowledge_tokens)
            prefix_mask.append(~knowledge_mask)
        hidden = torch.cat((*prefix, tokens), dim=1)
        padding_mask = torch.cat(
            (*prefix_mask, torch.zeros_like(values, dtype=torch.bool)), dim=1
        )
        if route_anchor is None:
            token_mean = tokens.mean(dim=1)
            if knowledge_tokens is None:
                route_anchor = cls[:, 0] + token_mean
            else:
                denominator = knowledge_mask.sum(dim=1, keepdim=True).clamp_min(1)
                prior_mean = (knowledge_tokens * knowledge_mask.unsqueeze(-1)).sum(dim=1)
                route_anchor = cls[:, 0] + token_mean + prior_mean / denominator
        encoded = self.transformer(hidden, padding_mask, route_anchor)
        prefix_length = hidden.shape[1] - tokens.shape[1]
        return CellViewOutput(
            cls=encoded.hidden[:, 0],
            token_hidden=encoded.hidden[:, prefix_length:],
            gene_indices=selected_indices,
            route_anchor=route_anchor,
            diagnostics=encoded.diagnostics,
        )


class FiLMExpressionDecoder(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.film = nn.Linear(dimension, dimension * 2)
        self.output = nn.Linear(dimension, 1)

    def forward(self, cls: torch.Tensor, gene_embeddings: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cls).chunk(2, dim=-1)
        conditioned = gene_embeddings.unsqueeze(0) * (1.0 + scale.unsqueeze(1))
        conditioned = conditioned + shift.unsqueeze(1)
        return self.output(torch.nn.functional.gelu(conditioned)).squeeze(-1)


class DinoGenePT(nn.Module):
    def __init__(
        self,
        *,
        n_genes: int,
        prior_dimensions: dict[str, int],
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.config = config
        dimension = int(config["d_model"])
        self.backbone = CellBackbone(n_genes, config)
        self.prior_adapters = nn.ModuleDict(
            {
                source: nn.Sequential(
                    nn.LayerNorm(prior_dim),
                    nn.Linear(prior_dim, dimension),
                    nn.GELU(),
                    nn.Linear(dimension, dimension),
                )
                for source, prior_dim in prior_dimensions.items()
            }
        )
        dino = config["dino_head"]
        self.dino_head = DINOHead(
            dimension,
            int(dino["hidden_dim"]),
            int(dino["bottleneck_dim"]),
            int(dino["prototypes"]),
        )
        self.expression_decoder = FiLMExpressionDecoder(dimension)
        self.token_delta_head = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, 1)
        )

    def adapt_prior(self, source: str, vectors: torch.Tensor) -> torch.Tensor:
        return self.prior_adapters[source](vectors)

    def conditional_view(
        self,
        control_expression: torch.Tensor,
        *,
        source: str,
        prior_vectors: torch.Tensor,
        prior_mask: torch.Tensor,
        route_anchor: torch.Tensor | None = None,
        gene_indices: torch.Tensor | None = None,
    ) -> CellViewOutput:
        return self.backbone(
            control_expression,
            knowledge_tokens=self.adapt_prior(source, prior_vectors),
            knowledge_mask=prior_mask,
            gene_indices=gene_indices,
            route_anchor=route_anchor,
        )

    def observed_view(
        self,
        post_expression: torch.Tensor,
        *,
        gene_indices: torch.Tensor | None = None,
        expression_mask: torch.Tensor | None = None,
    ) -> CellViewOutput:
        return self.backbone(
            post_expression,
            gene_indices=gene_indices,
            expression_mask=expression_mask,
        )

    def predict_expression(
        self, control_expression: torch.Tensor, conditional: CellViewOutput
    ) -> torch.Tensor:
        delta = self.expression_decoder(
            conditional.cls, self.backbone.gene_embedding.weight
        )
        return control_expression + delta

    def load_pretrained(
        self, path: str | Path, *, minimum_match_fraction: float
    ) -> dict[str, Any]:
        if not 0 < minimum_match_fraction <= 1:
            raise ValueError("pretrained minimum match fraction must be in (0, 1]")
        source = Path(path).resolve(strict=True)
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("pretrained checkpoint must contain a state dictionary")
        state = payload
        for key in ("model", "state_dict", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                state = payload[key]
                break
        normalized = {
            str(name).removeprefix("module."): value
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        current = self.state_dict()
        compatible = {
            name: value
            for name, value in normalized.items()
            if name in current and current[name].shape == value.shape
        }
        checkpoint_parameters = sum(value.numel() for value in normalized.values())
        matched_parameters = sum(value.numel() for value in compatible.values())
        match_fraction = (
            matched_parameters / checkpoint_parameters if checkpoint_parameters else 0.0
        )
        if match_fraction < minimum_match_fraction:
            raise ValueError(
                "pretrained checkpoint compatibility is below the configured gate: "
                f"matched={match_fraction:.4f}, required={minimum_match_fraction:.4f}"
            )
        result = self.load_state_dict(compatible, strict=False)
        return {
            "path": str(source),
            "checkpoint_tensor_count": len(normalized),
            "matched_tensor_count": len(compatible),
            "matched_parameter_fraction": match_fraction,
            "minimum_match_fraction": minimum_match_fraction,
            "missing_keys": list(result.missing_keys),
            "unexpected_keys": sorted(set(normalized) - set(compatible)),
        }

    def parameter_receipt(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total": total, "trainable": trainable}


class EMATeacher(nn.Module):
    def __init__(self, student: DinoGenePT) -> None:
        super().__init__()
        self.backbone = copy.deepcopy(student.backbone)
        self.dino_head = copy.deepcopy(student.dino_head)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def update(self, student: DinoGenePT, momentum: float) -> None:
        for teacher_module, student_module in (
            (self.backbone, student.backbone),
            (self.dino_head, student.dino_head),
        ):
            student_parameters = dict(student_module.named_parameters())
            for name, target in teacher_module.named_parameters():
                target.mul_(momentum).add_(student_parameters[name], alpha=1.0 - momentum)
            student_buffers = dict(student_module.named_buffers())
            for name, target in teacher_module.named_buffers():
                if name in student_buffers:
                    target.copy_(student_buffers[name])

    @torch.no_grad()
    def forward(
        self, post_expression: torch.Tensor, *, gene_indices: torch.Tensor | None = None
    ) -> tuple[CellViewOutput, torch.Tensor]:
        view = self.backbone(post_expression, gene_indices=gene_indices)
        return view, self.dino_head(view.cls)
