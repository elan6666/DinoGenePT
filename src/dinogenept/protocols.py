"""Runtime protocols that keep models separate from the common evaluator."""

from __future__ import annotations

from typing import Any, Protocol


class ModelPlugin(Protocol):
    def build_model(
        self,
        config: dict[str, Any],
        *,
        n_genes: int,
        prior_dimensions: dict[str, int],
    ) -> Any: ...

    def build_trainer(
        self,
        *,
        model: Any,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
    ) -> Any: ...

    def predict_split(
        self,
        *,
        model: Any,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
        split: str,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]: ...

    def permutation_check(
        self,
        *,
        model: Any,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
    ) -> dict[str, float | bool]: ...
