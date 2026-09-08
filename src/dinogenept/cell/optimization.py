"""Shared optimizer policy; gene identity embeddings never receive weight decay.

CellFM utils.set_weight_decay excludes embedding parameters. We target native
nn.Embedding modules explicitly rather than relying on parameter-name substrings.
Other parameter groups retain our existing policy. Zero decay does not erase
Adam momentum for a previously observed gene.
"""

from torch import nn


def optimizer_groups(model, weight_decay):
    embeddings = {id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)}
    decay, no_decay = [], []
    for parameter in model.parameters():
        if parameter.requires_grad:
            (no_decay if id(parameter) in embeddings else decay).append(parameter)
    return [dict(params=params, weight_decay=wd) for params, wd in
            ((decay, weight_decay), (no_decay, 0.0)) if params]
