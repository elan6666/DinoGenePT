"""Model-independent perturbation evaluation."""

from __future__ import annotations

from typing import Any

from .evaluator import PerturbationEvaluator
from .gradpert import GradPertEvaluator


def build_evaluator(config: dict[str, Any]) -> PerturbationEvaluator:
    return PerturbationEvaluator(config["evaluation"])


__all__ = ["GradPertEvaluator", "PerturbationEvaluator", "build_evaluator"]
