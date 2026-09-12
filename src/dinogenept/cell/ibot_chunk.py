"""Exact per-cell iBOT with checkpointed head+CE, not a logit concatenation.

Independent implementation: recompute both head logits/soft targets during
backward. Only compact features, weights and a frozen center survive forward.
No collectives or center mutations occur inside checkpointed closures.
"""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def projected_ibot(student_head, teacher_head, student, teacher, weights, center,
                   teacher_temperature, student_temperature, chunk_size=0):
    if chunk_size < 0 or student.shape != teacher.shape or student.ndim != 2:
        raise ValueError("Invalid iBOT chunk or feature axes")
    if weights.shape != student.shape[:1] or min(teacher_temperature, student_temperature) <= 0:
        raise ValueError("Invalid iBOT weights/temperatures")
    frozen_center = center.detach().float().clone()

    def part(s, t, w):
        with torch.no_grad():
            target_logits = teacher_head(t).float()
            target = ((target_logits - frozen_center) / teacher_temperature).softmax(-1)
            total = target_logits.sum(0)
        prediction = student_head(s).float() / student_temperature
        loss = (-(target * F.log_softmax(prediction, -1)).sum(-1) * w).sum()
        return loss, total

    n = student.shape[0]
    size = chunk_size or max(1, n)
    loss = student.sum() * 0
    total = frozen_center.new_zeros(frozen_center.shape)
    # Execute an empty head for zero-mask globals: separate-head DDP still sees
    # zero gradients rather than unused parameters. Range also covers n==0.
    for start in range(0, max(1, n), size):
        args = (student[start:start + size], teacher[start:start + size], weights[start:start + size])
        value, subtotal = (checkpoint(part, *args, use_reentrant=False)
                           if chunk_size and torch.is_grad_enabled() else part(*args))
        loss = loss + value
        total = total + subtotal.detach()
    return loss, total, total.new_tensor(float(n))
