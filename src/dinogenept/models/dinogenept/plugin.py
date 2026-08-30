"""DinoGenePT-specific training and inference behind the common model protocol."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

from dinogenept.experiments.training import Trainer

from .model import DinoGenePT, EMATeacher


def _condition_targets(data: Any) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for condition, targets in zip(data.conditions, data.targets, strict=True):
        if condition != "ctrl":
            result.setdefault(condition, targets)
    return result


class DinoGenePTPlugin:
    def build_model(
        self,
        config: dict[str, Any],
        *,
        n_genes: int,
        prior_dimensions: dict[str, int],
    ) -> DinoGenePT:
        return DinoGenePT(
            n_genes=n_genes,
            prior_dimensions=prior_dimensions,
            config={**config["model"], "ablation": config["ablation"]},
        )

    def build_trainer(
        self,
        *,
        model: DinoGenePT,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
    ) -> Trainer:
        return Trainer(
            model=model,
            teacher=EMATeacher(model).to(device),
            data=data,
            priors=priors,
            config=config,
            device=device,
        )

    def predict_split(
        self,
        *,
        model: DinoGenePT,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
        split: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
        evaluation = config["evaluation"]
        conditions = data.split_conditions(split)
        if not conditions:
            raise ValueError(f"dataset has no {split} conditions")
        control_indices = np.flatnonzero(data.control_mask)
        targets_by_condition = _condition_targets(data)
        samples = int(evaluation.get("control_samples", 300))
        batch_size = int(evaluation.get("batch_size", 64))
        rng = np.random.default_rng(int(evaluation.get("seed", 1)))
        predictions: dict[str, np.ndarray] = {}
        truths: dict[str, np.ndarray] = {}
        controls: dict[str, np.ndarray] = {}
        use_amp = bool(config["training"].get("amp_bfloat16", True)) and device.type == "cuda"
        model.eval()
        with torch.no_grad():
            for condition in conditions:
                selected_controls = data.expression[
                    rng.choice(control_indices, samples, replace=True)
                ].astype(np.float32)
                condition_predictions = []
                for start in range(0, samples, batch_size):
                    current = selected_controls[start : start + batch_size]
                    current_targets = tuple(
                        targets_by_condition[condition] for _ in range(len(current))
                    )
                    vectors, token_mask, available = priors.batch("base", current_targets)
                    if not available.all():
                        raise RuntimeError("Base prior disappeared during evaluation")
                    control_tensor = torch.as_tensor(current, device=device)
                    context = (
                        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if use_amp
                        else nullcontext()
                    )
                    with context:
                        view = model.conditional_view(
                            control_tensor,
                            source="base",
                            prior_vectors=torch.as_tensor(vectors, device=device),
                            prior_mask=torch.as_tensor(token_mask, device=device),
                        )
                        prediction = model.predict_expression(control_tensor, view)
                    condition_predictions.append(prediction.float().cpu().numpy())
                predictions[condition] = np.concatenate(condition_predictions, axis=0)
                truths[condition] = data.expression[data.indices_for_condition(condition)]
                controls[condition] = selected_controls
        return predictions, truths, controls

    def permutation_check(
        self,
        *,
        model: DinoGenePT,
        data: Any,
        priors: Any,
        config: dict[str, Any],
        device: Any,
    ) -> dict[str, float | bool]:
        targets_by_condition = _condition_targets(data)
        condition = data.split_conditions("train")[0]
        control = torch.as_tensor(
            data.expression[np.flatnonzero(data.control_mask)[:2]], device=device
        )
        condition_targets = tuple(
            targets_by_condition[condition] for _ in range(len(control))
        )
        vectors, token_mask, _ = priors.batch("base", condition_targets)
        prior = torch.as_tensor(vectors, device=device)
        mask = torch.as_tensor(token_mask, device=device)
        with torch.no_grad():
            gene_indices, _ = model.backbone.select_genes(control)
            baseline = model.conditional_view(
                control,
                source="base",
                prior_vectors=prior,
                prior_mask=mask,
                gene_indices=gene_indices,
            )
            order = torch.randperm(gene_indices.shape[1], device=device)
            permuted = model.conditional_view(
                control,
                source="base",
                prior_vectors=prior,
                prior_mask=mask,
                gene_indices=gene_indices[:, order],
            )
        difference = (baseline.cls - permuted.cls).abs()
        tolerance = float(config["runtime"].get("permutation_tolerance", 1e-5))
        return {
            "max_abs_difference": float(difference.max().cpu()),
            "mean_abs_difference": float(difference.mean().cpu()),
            "tolerance": tolerance,
            "passed": bool(float(difference.max().cpu()) <= tolerance),
        }


PLUGIN = DinoGenePTPlugin()
