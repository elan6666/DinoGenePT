"""Native reduced Kimi-inspired cell backbone; no upstream model imports.

See NATIVE_SOURCE_LEDGER.md for fidelity/adaptations. The scale, initialization,
gene masks and noncausal global layers are project choices, not Kimi weights.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .kda import batched_chunk_kda, chunk_kda, parallel_chunk_kda


@dataclass(frozen=True)
class BackboneConfig:
    genes: int
    width: int = 768
    depth: int = 12
    kda_heads: int = 6
    kda_head_dim: int = 128
    mla_heads: int = 12
    mla_head_dim: int = 64
    mla_shared_dim: int = 32
    query_rank: int = 192
    kv_rank: int = 128
    global_every: int = 4
    residual_block_size: int = 4
    ffn_expansion: int = 1
    chunk_size: int = 16
    kda_implementation: str = "chunk"
    gradient_checkpointing: bool = True
    norm_eps: float = 1e-5
    expression_basis: int = 256
    mixer_type: str = "hybrid_kda"
    residual_type: str = "attnres"
    ffn_type: str = "situ"
    kda_direction: str = "forward"
    short_convolution: int = 0
    delta_rule: str = "channel"
    global_attention: str = "mla"
    gqa_kv_heads: int = 1

    def __post_init__(self):
        values = (
            self.genes,
            self.width,
            self.depth,
            self.kda_heads,
            self.kda_head_dim,
            self.mla_heads,
            self.mla_head_dim,
            self.mla_shared_dim,
            self.query_rank,
            self.kv_rank,
            self.global_every,
            self.residual_block_size,
            self.ffn_expansion,
            self.chunk_size,
            self.expression_basis,
        )
        if any(x < 1 for x in values) or self.depth % self.global_every:
            raise ValueError("Invalid architecture dimensions; last layer must be global")
        if self.kda_implementation not in {"chunk", "parallel_chunk", "batched_chunk"}:
            raise ValueError("Unknown native KDA execution implementation")
        if self.mixer_type not in {"hybrid_kda", "full_mla", "eretnet"}:
            raise ValueError("Unknown mixer_type")
        if self.residual_type not in {"attnres", "standard", "deepnorm"}:
            raise ValueError("Unknown residual_type")
        if self.ffn_type not in {"situ", "swiglu", "sglu"}:
            raise ValueError("Unknown ffn_type")
        if self.kda_direction not in {"forward", "bidirectional"} or self.short_convolution not in {0, 4}:
            raise ValueError("Invalid KDA direction/convolution")
        if self.mixer_type == "eretnet" and self.width % self.mla_heads:
            raise ValueError("ERetNet width must divide heads")
        if self.delta_rule not in {"channel", "scalar"} or self.global_attention not in {"mla", "gqa"}:
            raise ValueError("Unknown delta/global attention mechanism")
        if self.gqa_kv_heads < 1 or self.mla_heads % self.gqa_kv_heads:
            raise ValueError("GQA query heads must divide KV heads")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight.float()).to(x.dtype)


class ValueEncoder(nn.Module):
    def __init__(self, width, basis=256):
        super().__init__()
        self.input = nn.Linear(1, basis, bias=False)
        self.mix = nn.Linear(basis, basis, bias=False)
        self.table = nn.Linear(basis, width, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.mask = nn.Parameter(torch.zeros(width))

    def forward(self, values, hidden):
        # Never process hidden numerical values, even if a caller passes NaN.
        x = torch.where(hidden, 0, values).unsqueeze(-1)
        x = F.leaky_relu(self.input(x), negative_slope=0.2)
        x = self.table((self.mix(x) + self.alpha * x).softmax(-1))
        return torch.where(hidden.unsqueeze(-1), self.mask, x)


class KDAMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.kda_heads, config.kda_head_dim
        size = self.heads * self.dim
        self.q_proj = nn.Linear(config.width, size, bias=False)
        self.k_proj = nn.Linear(config.width, size, bias=False)
        self.v_proj = nn.Linear(config.width, size, bias=False)
        self.scalar_decay = config.delta_rule == "scalar"
        if self.scalar_decay:
            self.decay_proj = nn.Linear(config.width, self.heads, bias=False)
        else:
            self.f_a_proj = nn.Linear(config.width, self.dim, bias=False)
            self.f_b_proj = nn.Linear(self.dim, size, bias=False)
        self.b_proj = nn.Linear(config.width, self.heads, bias=False)
        self.g_proj = nn.Linear(config.width, size, bias=False)
        self.o_norm = RMSNorm(self.dim, config.norm_eps)
        self.o_proj = nn.Linear(size, config.width, bias=False)
        self.a_log = nn.Parameter(torch.empty(self.heads).uniform_(1, 16).log())
        # HF inference source allocates empty dt_bias expecting checkpoint data.
        # Our from-scratch initializer targets log decay -0.1 at zero projection.
        target_logit = torch.logit(torch.tensor(0.1 / 5))
        if self.scalar_decay:
            self.dt_bias = nn.Parameter(torch.log(torch.expm1(.1 / self.a_log.detach().exp())))
        else:
            self.dt_bias = nn.Parameter((target_logit / self.a_log.detach().exp()).repeat_interleave(self.dim))
        self.chunk_size = config.chunk_size
        self.direction = config.kda_direction
        self.convs = (nn.ModuleDict({name: nn.Conv1d(size, size, 4, groups=size, bias=False)
                                    for name in ("q", "k", "v")}) if config.short_convolution else None)
        self.kernel = {"chunk": chunk_kda, "parallel_chunk": parallel_chunk_kda, "batched_chunk": batched_chunk_kda}[
            config.kda_implementation
        ]

    def forward(self, x, valid, writable):
        shape = (*x.shape[:2], self.heads, self.dim)
        # No short convolution. Retain the reference convolution's SiLU nonlinearity.
        def project(layer, name):
            value = layer(x)
            if self.convs is not None:
                # Mask before convolution: hidden tokens must not leak into
                # writable neighbours. Identity context remains at the query.
                context = value * writable.unsqueeze(-1)
                value = self.convs[name](F.pad(context.transpose(1, 2), (3, 0))).transpose(1, 2)
            return F.silu(value).reshape(shape)
        q = F.normalize(project(self.q_proj, "q").float(), dim=-1, eps=1e-6)
        k = F.normalize(project(self.k_proj, "k").float(), dim=-1, eps=1e-6)
        v = project(self.v_proj, "v")
        if self.scalar_decay:
            decay = -self.a_log.float().exp() * F.softplus(self.decay_proj(x).float() + self.dt_bias.float())
            decay = decay.unsqueeze(-1).expand(shape)
        else:
            decay_input = self.f_b_proj(self.f_a_proj(x)).reshape(shape).float()
            decay = -5 * torch.sigmoid(
                self.a_log.float().exp()[:, None] * (decay_input + self.dt_bias.float().reshape(self.heads, self.dim))
            )
        decay = torch.where(writable[:, :, None, None], decay, 0)
        beta = self.b_proj(x).float().sigmoid() * writable.unsqueeze(-1)
        output, _ = self.kernel(q, k, v, decay, beta, chunk_size=self.chunk_size)
        if self.direction == "bidirectional":
            reverse, _ = self.kernel(*(a.flip(1) for a in (q, k, v, decay, beta)), chunk_size=self.chunk_size)
            output = (output + reverse.flip(1)) * .5
        gate_logits = self.g_proj(x).reshape(shape).float()
        gate = F.silu(gate_logits) if self.scalar_decay else gate_logits.sigmoid()
        output = (self.o_norm(output).float() * gate).to(x.dtype).flatten(-2)
        return self.o_proj(output) * valid.unsqueeze(-1)


class GatedMLA(nn.Module):
    """Noncausal global cell attention; NoPE still retains shared key channels."""

    def __init__(self, config):
        super().__init__()
        self.heads, self.dim, self.shared = config.mla_heads, config.mla_head_dim, config.mla_shared_dim
        self.kv_rank = config.kv_rank
        self.q_a_proj = nn.Linear(config.width, config.query_rank, bias=False)
        self.q_norm = RMSNorm(config.query_rank, config.norm_eps)
        self.q_b_proj = nn.Linear(config.query_rank, self.heads * (self.dim + self.shared), bias=False)
        self.kv_a_proj = nn.Linear(config.width, self.kv_rank + self.shared, bias=False)
        self.kv_norm = RMSNorm(self.kv_rank, config.norm_eps)
        self.kv_b_proj = nn.Linear(self.kv_rank, self.heads * self.dim * 2, bias=False)
        self.g_proj = nn.Linear(config.width, self.heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, config.width, bias=False)
        self.qk_clip_threshold = None
        self.register_buffer("qk_max", torch.zeros(self.heads), persistent=False)

    def forward(self, x, valid, writable):
        b, t, _ = x.shape
        q = self.q_b_proj(self.q_norm(self.q_a_proj(x))).reshape(b, t, self.heads, self.dim + self.shared)
        latent, shared = self.kv_a_proj(x).split([self.kv_rank, self.shared], dim=-1)
        key, value = self.kv_b_proj(self.kv_norm(latent)).reshape(b, t, self.heads, 2 * self.dim).chunk(2, -1)
        key = torch.cat((key, shared.unsqueeze(-2).expand(-1, -1, self.heads, -1)), dim=-1)
        if self.training and self.qk_clip_threshold is not None:
            with torch.no_grad():
                # Block queries for audit statistics, never allocate B*H*T*T.
                k = key.detach().float().transpose(1, 2).transpose(-1, -2)
                for start in range(0, t, 64):
                    logits = q[:, start:start + 64].detach().float().transpose(1, 2) @ k
                    logits /= (self.dim + self.shared) ** .5
                    eligible = valid[:, None, start:start + 64, None] & writable[:, None, None, :]
                    logits.masked_fill_(~eligible, -torch.inf)
                    self.qk_max.copy_(torch.maximum(self.qk_max, logits.amax((0, 2, 3))))
        out = (
            F.scaled_dot_product_attention(
                q.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=writable[:, None, None, :],
                dropout_p=0,
                is_causal=False,
            )
            .transpose(1, 2)
            .reshape(b, t, -1)
        )
        return self.o_proj(out * self.g_proj(x).sigmoid()) * valid.unsqueeze(-1)


class GatedGQA(nn.Module):
    """NoPE/noncausal reduced grouped attention; not a Qwen checkpoint port."""

    def __init__(self, config):
        super().__init__()
        self.heads, self.kv_heads, self.dim = config.mla_heads, config.gqa_kv_heads, config.mla_head_dim
        self.q_proj = nn.Linear(config.width, self.heads * self.dim, bias=False)
        self.k_proj = nn.Linear(config.width, self.kv_heads * self.dim, bias=False)
        self.v_proj = nn.Linear(config.width, self.kv_heads * self.dim, bias=False)
        self.g_proj = nn.Linear(config.width, self.heads * self.dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.dim, config.width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.dim, config.norm_eps), RMSNorm(self.dim, config.norm_eps)

    def forward(self, x, valid, writable):
        b, t, _ = x.shape
        q = self.q_norm(self.q_proj(x).reshape(b, t, self.heads, self.dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).reshape(b, t, self.kv_heads, self.dim)).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.kv_heads, self.dim).transpose(1, 2)
        # Explicit repeat is portable; fused enable_gqa is a future perf option.
        repeats = self.heads // self.kv_heads
        out = F.scaled_dot_product_attention(q, k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1),
                                            attn_mask=writable[:, None, None, :], dropout_p=0.)
        out = out.transpose(1, 2).reshape(b, t, -1) * F.silu(self.g_proj(x))
        return self.o_proj(out) * valid.unsqueeze(-1)


class ERetMixer(nn.Module):
    """CellFM MHRetention algebra, independently expressed in torch.

    ReLU Q/K, divide each by sqrt(head_dim), noncausal Q(K^T V),
    scale-free RMS and SiLU output gate. No position decay or softmax.
    Init/mask/outer residual remain explicit project comparison choices.
    """

    def __init__(self, config):
        super().__init__()
        self.heads, self.dim = config.mla_heads, config.width // config.mla_heads
        for name in ("q_proj", "k_proj", "v_proj", "u_proj", "o_proj"):
            setattr(self, name, nn.Linear(config.width, config.width, bias=False))

    def forward(self, x, valid, writable):
        shape = (*x.shape[:2], self.heads, self.dim)
        q = self.q_proj(x).reshape(shape).transpose(1, 2).float().relu() / self.dim ** .5
        k = self.k_proj(x).reshape(shape).transpose(1, 2).float().relu() / self.dim ** .5
        k = k * writable[:, None, :, None]
        v = self.v_proj(x).reshape(shape).transpose(1, 2).float()
        output = q @ (k.transpose(-1, -2) @ v)
        output = output / (output.norm(dim=-1, keepdim=True) / self.dim ** .5).clamp_min(1e-12)
        gate = F.silu(self.u_proj(x).reshape(shape).transpose(1, 2).float())
        output = (output * gate).transpose(1, 2).flatten(-2).to(x.dtype)
        return self.o_proj(output) * valid.unsqueeze(-1)


class SiTUFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = config.width * config.ffn_expansion
        self.gate_proj = nn.Linear(config.width, hidden, bias=False)
        self.up_proj = nn.Linear(config.width, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, config.width, bias=False)
        self.kind = config.ffn_type

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.kind == "swiglu":
            return self.down_proj(F.silu(gate) * up)
        if self.kind == "sglu":
            return self.down_proj(gate * up)
        stable_gate = 4 * torch.tanh(gate.float() / 4) * gate.float().sigmoid()
        stable_up = 25 * torch.tanh(up.float() / 25)
        return self.down_proj((stable_gate * stable_up).to(x.dtype))


class DepthRead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.width, config.norm_eps)
        self.query = nn.Parameter(torch.zeros(config.width))

    def forward(self, prefix, completed):
        values = torch.stack((*completed, prefix), dim=-2)
        scores = (self.norm(values).float() * self.query.float()).sum(-1)
        return (scores.softmax(-1).unsqueeze(-1) * values.float()).sum(-2).to(prefix.dtype)


class CellBlock(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        global_cls = GatedMLA if config.global_attention == "mla" else GatedGQA
        self.mixer = (ERetMixer(config) if config.mixer_type == "eretnet" else
                      global_cls(config) if config.mixer_type == "full_mla" or (index + 1) % config.global_every == 0
                      else KDAMixer(config))
        # At the first mixer there is no depth history to retrieve. Do not
        # register dead query/norm parameters that can never receive gradients.
        self.mixer_read = DepthRead(config) if index and config.residual_type == "attnres" else None
        self.ffn_read = DepthRead(config) if config.residual_type == "attnres" else None
        norm = nn.LayerNorm if config.residual_type == "deepnorm" else RMSNorm
        self.mixer_norm, self.ffn_norm = norm(config.width, config.norm_eps), norm(config.width, config.norm_eps)
        self.ffn = SiTUFeedForward(config)


class CellBackbone(nn.Module):
    def __init__(self, config: BackboneConfig):
        super().__init__()
        self.config = config
        self.gene = nn.Embedding(config.genes + 1, config.width, padding_idx=0)
        self.value = ValueEncoder(config.width, basis=config.expression_basis)
        self.cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.blocks = nn.ModuleList(CellBlock(config, i) for i in range(config.depth))
        self.output_read = DepthRead(config) if config.residual_type == "attnres" else None
        self.norm = RMSNorm(config.width, config.norm_eps)
        self.apply(self._initialize)
        nn.init.normal_(self.cls, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def forward(self, gene_ids, expression, valid, hidden=None, prefix_tokens=None):
        if gene_ids.shape != expression.shape or valid.shape != gene_ids.shape or valid.dtype != torch.bool:
            raise ValueError("Gene/expression/valid axes must match")
        hidden = torch.zeros_like(valid) if hidden is None else hidden
        if hidden.shape != valid.shape or hidden.dtype != torch.bool:
            raise ValueError("Hidden mask axes/type mismatch")
        hidden = hidden & valid
        x = self.gene(gene_ids) + self.value(torch.where(valid, expression, 0), hidden)
        x = x * valid.unsqueeze(-1)
        b = x.shape[0]
        prefixes = [self.cls.expand(b, -1, -1)]
        if prefix_tokens is not None:
            if prefix_tokens.ndim != 3 or prefix_tokens.shape[0] != b or prefix_tokens.shape[-1] != self.config.width:
                raise ValueError("Invalid conditioning prefix")
            prefixes.append(prefix_tokens)
        prefix = torch.cat((*prefixes, x), dim=1)
        n_prefix = sum(item.shape[1] for item in prefixes)
        all_valid = torch.cat((valid.new_ones(b, n_prefix), valid), dim=1)
        early_write = torch.cat((valid.new_ones(b, n_prefix), valid & ~hidden), dim=1)
        completed = []
        for i, block in enumerate(self.blocks):
            if self.config.residual_type != "attnres":
                writable = early_write if i < self.config.depth // 2 else all_valid
                deep = self.config.residual_type == "deepnorm"
                alpha = (2 * self.config.depth) ** .25 if deep else 1.
                read = prefix if deep else block.mixer_norm(prefix)
                mixed = (checkpoint(block.mixer, read, all_valid, writable, use_reentrant=False)
                         if self.training and self.config.gradient_checkpointing
                         else block.mixer(read, all_valid, writable))
                prefix = alpha * prefix + mixed
                if deep:
                    prefix = block.mixer_norm(prefix)
                read = prefix if deep else block.ffn_norm(prefix)
                ffn = (checkpoint(block.ffn, read, use_reentrant=False)
                       if self.training and self.config.gradient_checkpointing else block.ffn(read))
                prefix = alpha * prefix + ffn
                if deep:
                    prefix = block.ffn_norm(prefix)
                prefix = prefix * all_valid.unsqueeze(-1)
                continue
            read = block.mixer_read(prefix, completed) if completed else prefix
            if i % self.config.residual_block_size == 0:
                completed.append(prefix)
                prefix = None
            writable = early_write if i < self.config.depth // 2 else all_valid
            read = block.mixer_norm(read)
            if self.training and self.config.gradient_checkpointing:
                mixed = checkpoint(block.mixer, read, all_valid, writable, use_reentrant=False)
            else:
                mixed = block.mixer(read, all_valid, writable)
            prefix = mixed if prefix is None else prefix + mixed
            read = block.ffn_norm(block.ffn_read(prefix, completed))
            ffn = (
                checkpoint(block.ffn, read, use_reentrant=False)
                if self.training and self.config.gradient_checkpointing
                else block.ffn(read)
            )
            prefix = (prefix + ffn) * all_valid.unsqueeze(-1)
        result = self.norm(self.output_read(prefix, completed) if self.output_read is not None else prefix)
        result = result * all_valid.unsqueeze(-1)
        return {"cls": result[:, 0], "genes": result[:, n_prefix:], "gene_identity": self.gene(gene_ids)}

    def lora_targets(self):
        paths = []
        for i, block in enumerate(self.blocks):
            leaves = (
                ("q_proj", "k_proj", "v_proj", "o_proj")
                if isinstance(block.mixer, (KDAMixer, ERetMixer, GatedGQA))
                else ("q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj", "o_proj")
            )
            paths.extend(f"blocks.{i}.mixer.{name}" for name in leaves)
        return paths
