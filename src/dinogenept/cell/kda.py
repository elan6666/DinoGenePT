"""Native single-direction KDA numerical kernels.

The recurrent equation is source-checked against FLA's reference (see ledger).
The chunk implementation solves the within-chunk lower-triangular delta system
using PyTorch; it does not call FLA or claim parity with its fused CUDA kernel.
Both accept pre-normalized q/k, log decay, and sigmoid beta. Float32 state.
"""

import torch


def _validate(q, k, v, log_decay, beta):
    if q.ndim != 4 or k.shape != q.shape or log_decay.shape != q.shape:
        raise ValueError("q/k/log_decay must share [batch,time,head,key] shape")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3] or beta.shape != q.shape[:3]:
        raise ValueError("v/beta axes mismatch; grouped-value heads are not supported")
    if q.shape[1] == 0:
        raise ValueError("KDA requires a nonempty sequence")


def recurrent_kda(q, k, v, log_decay, beta, initial_state=None):
    """Small-sequence reference, not the default long-sequence training kernel."""
    _validate(q, k, v, log_decay, beta)
    dtype = v.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = [x.float() for x in (q, k, v, log_decay, beta)]
        b, t, h, d = q.shape
        state = q.new_zeros(b, h, d, v.shape[-1]) if initial_state is None else initial_state.float()
        if state.shape != (b, h, d, v.shape[-1]):
            raise ValueError("Initial state shape mismatch")
        outputs = []
        for index in range(t):
            state = state * g[:, index].exp().unsqueeze(-1)
            residual = v[:, index] - (state * k[:, index].unsqueeze(-1)).sum(-2)
            state = state + (beta[:, index, :, None] * k[:, index]).unsqueeze(-1) * residual.unsqueeze(-2)
            outputs.append((state * (q[:, index] * d**-0.5).unsqueeze(-1)).sum(-2))
        return torch.stack(outputs, dim=1).to(dtype), state


def chunk_kda(q, k, v, log_decay, beta, initial_state=None, *, chunk_size=16):
    """Differentiable chunkwise triangular solve; no T-by-T attention allocation.

    For i>=j within one chunk, D_ij=exp(G_i-G_j), with G cumulative log decay.
    Solve (I+beta_i * tril(<k_i,D_ij*k_j>,-1)) u
        = beta_i*(v_i - <k_i*exp(G_i), S_start>).
    Read q_i from S_start plus the causal sum of delta updates u_j.
    Caller must supply nonpositive log decays. Bounded gates prevent growth.
    """
    _validate(q, k, v, log_decay, beta)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    dtype = v.dtype
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, g, beta = [x.float().transpose(1, 2) for x in (q, k, v, log_decay, beta)]
        b, h, t, d = q.shape
        state = q.new_zeros(b, h, d, v.shape[-1]) if initial_state is None else initial_state.float()
        if state.shape != (b, h, d, v.shape[-1]):
            raise ValueError("Initial state shape mismatch")
        outputs = []
        for start in range(0, t, chunk_size):
            end = min(t, start + chunk_size)
            qi, ki, vi = q[:, :, start:end] * d**-0.5, k[:, :, start:end], v[:, :, start:end]
            bi, gi = beta[:, :, start:end], g[:, :, start:end].cumsum(-2)
            n = end - start
            causal = torch.ones(n, n, device=q.device, dtype=torch.bool).tril()
            differences = gi.unsqueeze(-2) - gi.unsqueeze(-3)
            # Mask BEFORE exp: discarded future entries may otherwise overflow.
            decay = differences.masked_fill(~causal.unsqueeze(-1), -torch.inf).exp()
            kk = torch.einsum("bhid,bhjd,bhijd->bhij", ki, ki, decay)
            system = (kk * bi.unsqueeze(-1)).tril(-1) + torch.eye(n, device=q.device)
            residual = (vi - (ki * gi.exp()) @ state) * bi.unsqueeze(-1)
            updates = torch.linalg.solve_triangular(system, residual, upper=False, unitriangular=True)
            qk = torch.einsum("bhid,bhjd,bhijd->bhij", qi, ki, decay)
            outputs.append((qi * gi.exp()) @ state + qk @ updates)
            ending_decay = (gi[:, :, -1:] - gi).exp()
            state = state * gi[:, :, -1].exp().unsqueeze(-1) + (ki * ending_decay).transpose(-1, -2) @ updates
        return torch.cat(outputs, dim=2).transpose(1, 2).to(dtype), state
