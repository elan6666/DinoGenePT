"""Factory for the single DinoGenePT model family."""

from __future__ import annotations

from typing import Any

from .model import DinoGenePT
from .plugin import PLUGIN


def build_model(
    config: dict[str, Any], *, n_genes: int, prior_dimensions: dict[str, int]
) -> DinoGenePT:
    return PLUGIN.build_model(
        config, n_genes=n_genes, prior_dimensions=prior_dimensions
    )


__all__ = ["DinoGenePT", "PLUGIN", "build_model"]
