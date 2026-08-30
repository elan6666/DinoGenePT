"""Transformer, KDA, attention-residual, and sparse-adapter modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class SiTUGLU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.up = nn.Linear(input_dim, hidden_dim)
        self.down = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        gate_projection = self.gate(inputs)
        gate = 4.0 * torch.tanh(gate_projection / 4.0) * torch.sigmoid(gate_projection)
        up = 25.0 * torch.tanh(self.up(inputs) / 25.0)
        return self.down(gate * up)


class FeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dim: int, dropout: float, *, situ: bool) -> None:
        super().__init__()
        if situ:
            self.network = SiTUGLU(dimension, hidden_dim, dimension)
        else:
            self.network = nn.Sequential(
                nn.Linear(dimension, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dimension),
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.network(inputs))


class FullSelfAttention(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        output, _ = self.attention(
            inputs,
            inputs,
            inputs,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        return output


class DeltaAttentionDirection(nn.Module):
    """Reference KDA recurrence; intended for correctness/smoke, not fused production use."""

    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("KDA heads must divide the model dimension")
        self.heads = heads
        self.head_dim = dimension // heads
        self.qkv = nn.Linear(dimension, dimension * 3, bias=False)
        self.alpha = nn.Linear(dimension, dimension)
        self.beta = nn.Linear(dimension, heads)
        self.output_gate = nn.Linear(dimension, dimension)
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.norm = nn.RMSNorm(self.head_dim)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, dimension = inputs.shape
        qkv = self.qkv(inputs).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = F.normalize(F.silu(query), dim=-1)
        key = F.normalize(F.silu(key), dim=-1)
        value = F.silu(value)
        alpha = torch.exp(-5.0 * torch.sigmoid(self.alpha(inputs))).view(
            batch, length, self.heads, self.head_dim
        )
        beta = torch.sigmoid(self.beta(inputs))
        state = inputs.new_zeros(batch, self.heads, self.head_dim, self.head_dim)
        outputs: list[torch.Tensor] = []
        for position in range(length):
            valid = (
                inputs.new_ones(batch, dtype=torch.bool)
                if padding_mask is None
                else ~padding_mask[:, position]
            )
            current_alpha = torch.where(
                valid[:, None, None], alpha[:, position], torch.ones_like(alpha[:, position])
            )
            current_beta = torch.where(
                valid[:, None], beta[:, position], torch.zeros_like(beta[:, position])
            )
            decayed = current_alpha.unsqueeze(-1) * state
            prediction = torch.einsum("bhde,bhd->bhe", decayed, key[:, position])
            error = value[:, position] - prediction
            update = torch.einsum("bhd,bhe->bhde", key[:, position], error)
            state = decayed + current_beta[:, :, None, None] * update
            current = torch.einsum("bhde,bhd->bhe", state, query[:, position])
            current = self.norm(current)
            outputs.append(current.reshape(batch, dimension))
        stacked = torch.stack(outputs, dim=1)
        gate = torch.sigmoid(self.output_gate(inputs))
        return self.output(gate * stacked)


class BidirectionalDeltaAttention(nn.Module):
    def __init__(self, dimension: int, heads: int) -> None:
        super().__init__()
        self.forward_direction = DeltaAttentionDirection(dimension, heads)
        self.backward_direction = DeltaAttentionDirection(dimension, heads)
        self.merge = nn.Linear(dimension * 2, dimension)

    def forward(self, inputs: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        forward = self.forward_direction(inputs, padding_mask)
        reversed_inputs = torch.flip(inputs, dims=(1,))
        reversed_mask = None if padding_mask is None else torch.flip(padding_mask, dims=(1,))
        backward = torch.flip(
            self.backward_direction(reversed_inputs, reversed_mask), dims=(1,)
        )
        return self.merge(torch.cat((forward, backward), dim=-1))


class BlockAttentionResidual(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.empty(dimension))
        self.gate = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.query, std=dimension**-0.5)
        self.norm = nn.RMSNorm(dimension)

    def forward(self, snapshots: list[torch.Tensor]) -> torch.Tensor:
        values = torch.stack(snapshots, dim=2)
        keys = self.norm(values)
        scores = torch.einsum("blnd,d->bln", keys, self.query) / math.sqrt(keys.shape[-1])
        weights = scores.softmax(dim=-1)
        retrieved = torch.einsum("bln,blnd->bld", weights, values)
        return torch.tanh(self.gate) * retrieved


class ResidualDenseAdapter(nn.Module):
    def __init__(self, dimension: int, latent_dim: int, dropout: float, *, situ: bool) -> None:
        super().__init__()
        self.down = nn.Linear(dimension, latent_dim)
        self.expert = FeedForward(latent_dim, latent_dim * 2, dropout, situ=situ)
        self.up = nn.Linear(latent_dim, dimension)
        self.norm = nn.RMSNorm(latent_dim)
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(
        self, inputs: torch.Tensor, route_anchor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        del route_anchor
        output = self.up(self.norm(self.expert(self.down(inputs))))
        return torch.tanh(self.scale) * output, {}


class ResidualLatentMoE(nn.Module):
    def __init__(
        self,
        dimension: int,
        latent_dim: int,
        experts: int,
        top_k: int,
        dropout: float,
        *,
        situ: bool,
        balance: str,
    ) -> None:
        super().__init__()
        if experts < 2 or not 0 < top_k <= experts:
            raise ValueError("invalid MoE expert count or top-k")
        self.expert_count = experts
        self.top_k = top_k
        self.balance = balance
        self.down = nn.Linear(dimension, latent_dim)
        self.shared = FeedForward(latent_dim, latent_dim * 2, dropout, situ=situ)
        self.experts = nn.ModuleList(
            FeedForward(latent_dim, latent_dim * 2, dropout, situ=situ)
            for _ in range(experts)
        )
        self.router = nn.Linear(dimension, experts)
        self.up = nn.Linear(latent_dim, dimension)
        self.norm = nn.RMSNorm(latent_dim)
        self.scale = nn.Parameter(torch.zeros(()))
        self.register_buffer("router_bias", torch.zeros(experts), persistent=True)

    @torch.no_grad()
    def _quantile_balance(self, raw_scores: torch.Tensor, cutoff: torch.Tensor) -> None:
        quantile = 1.0 - self.top_k / self.expert_count
        margins = raw_scores - cutoff[:, None]
        proposed = -torch.quantile(margins.float(), quantile, dim=0).to(self.router_bias)
        proposed -= proposed.mean()
        self.router_bias.mul_(0.9).add_(proposed, alpha=0.1)

    def forward(
        self, inputs: torch.Tensor, route_anchor: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        latent = self.down(inputs)
        raw_scores = torch.sigmoid(self.router(route_anchor))
        biased = raw_scores + self.router_bias
        expanded = torch.topk(biased, min(self.top_k + 1, self.expert_count), dim=-1)
        selected = expanded.indices[:, : self.top_k]
        selected_raw = raw_scores.gather(1, selected)
        weights = selected_raw / selected_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        routed = torch.zeros_like(latent)
        load = torch.zeros(self.expert_count, device=inputs.device, dtype=inputs.dtype)
        for expert_index, expert in enumerate(self.experts):
            locations = torch.nonzero(selected == expert_index, as_tuple=False)
            if not len(locations):
                continue
            batch_indices = locations[:, 0]
            slots = locations[:, 1]
            expert_inputs = latent.index_select(0, batch_indices)
            expert_output = expert(expert_inputs)
            expert_weights = weights[batch_indices, slots, None, None].to(
                dtype=expert_output.dtype
            )
            routed.index_add_(0, batch_indices, expert_output * expert_weights)
            load[expert_index] = len(batch_indices)
        if self.training and self.balance == "quantile" and expanded.values.shape[1] > self.top_k:
            self._quantile_balance(raw_scores.detach(), expanded.values[:, self.top_k].detach())
        probabilities = raw_scores / raw_scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        load_fraction = load / load.sum().clamp_min(1.0)
        importance = probabilities.mean(dim=0)
        balance_loss = self.expert_count * torch.sum(load_fraction * importance)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1).mean()
        combined = self.shared(latent) + routed
        output = self.up(self.norm(combined))
        return torch.tanh(self.scale) * output, {
            "moe_balance_loss": balance_loss,
            "moe_entropy": entropy,
            "moe_load": load,
        }


@dataclass
class EncoderOutput:
    hidden: torch.Tensor
    diagnostics: dict[str, torch.Tensor]


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        *,
        attention: str,
        situ: bool,
        adapter: nn.Module | None,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.attention = (
            BidirectionalDeltaAttention(dimension, heads)
            if attention == "kda"
            else FullSelfAttention(dimension, heads, dropout)
        )
        self.ff = FeedForward(dimension, ff_dim, dropout, situ=situ)
        self.dropout = nn.Dropout(dropout)
        self.adapter = adapter

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor | None,
        route_anchor: torch.Tensor,
    ) -> EncoderOutput:
        hidden = inputs + self.dropout(self.attention(self.norm1(inputs), padding_mask))
        hidden = hidden + self.ff(self.norm2(hidden))
        diagnostics: dict[str, torch.Tensor] = {}
        if self.adapter is not None:
            residual, diagnostics = self.adapter(hidden, route_anchor)
            hidden = hidden + residual
        return EncoderOutput(hidden=hidden, diagnostics=diagnostics)


class CellTransformer(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        dimension = int(config["d_model"])
        layers = int(config["layers"])
        heads = int(config["heads"])
        ff_dim = int(config["ff_dim"])
        dropout = float(config.get("dropout", 0.1))
        ablation = config["ablation"]
        kda = ablation.get("kda", {})
        adapter_config = ablation.get("adapter", {})
        adapter_start = max(0, layers - int(adapter_config.get("top_layers", 0)))
        blocks: list[TransformerBlock] = []
        for index in range(layers):
            attention = "full"
            if kda.get("enabled") and (index + 1) % int(kda.get("global_every", 4)) != 0:
                attention = "kda"
            adapter: nn.Module | None = None
            if index >= adapter_start and adapter_config.get("type") == "dense":
                adapter = ResidualDenseAdapter(
                    dimension,
                    int(adapter_config["latent_dim"]),
                    dropout,
                    situ=bool(adapter_config.get("situ_glu", False)),
                )
            elif index >= adapter_start and adapter_config.get("type") == "moe":
                adapter = ResidualLatentMoE(
                    dimension,
                    int(adapter_config["latent_dim"]),
                    int(adapter_config.get("experts", 4)),
                    int(adapter_config.get("top_k", 2)),
                    dropout,
                    situ=bool(adapter_config.get("situ_glu", False)),
                    balance=str(adapter_config.get("balance", "auxiliary")),
                )
            blocks.append(
                TransformerBlock(
                    dimension,
                    heads,
                    ff_dim,
                    dropout,
                    attention=attention,
                    situ=bool(ablation.get("situ_glu_backbone", False)),
                    adapter=adapter,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.attnres_enabled = bool(ablation.get("attention_residual", {}).get("enabled"))
        self.attnres_block_size = int(
            ablation.get("attention_residual", {}).get("block_size", 4)
        )
        retrieval_count = (
            (layers - 1) // self.attnres_block_size if self.attnres_enabled else 0
        )
        self.attnres = nn.ModuleList(
            BlockAttentionResidual(dimension) for _ in range(retrieval_count)
        )
        self.final_norm = nn.LayerNorm(dimension)

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: torch.Tensor | None,
        route_anchor: torch.Tensor,
    ) -> EncoderOutput:
        hidden = inputs
        snapshots = [inputs]
        diagnostics: dict[str, torch.Tensor] = {}
        for index, block in enumerate(self.blocks):
            if self.attnres_enabled and index and index % self.attnres_block_size == 0:
                hidden = hidden + self.attnres[
                    index // self.attnres_block_size - 1
                ](snapshots)
            output = block(hidden, padding_mask, route_anchor)
            hidden = output.hidden
            for key, value in output.diagnostics.items():
                diagnostics[f"layer_{index}.{key}"] = value
            if self.attnres_enabled and (index + 1) % self.attnres_block_size == 0:
                snapshots.append(hidden)
        return EncoderOutput(hidden=self.final_norm(hidden), diagnostics=diagnostics)


class DINOHead(nn.Module):
    def __init__(self, dimension: int, hidden: int, bottleneck: int, prototypes: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimension, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.prototypes = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck, prototypes, bias=False)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.prototypes(F.normalize(self.network(inputs), dim=-1))
