"""Condition-macro evaluator and paired-condition bootstrap summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import condition_metrics


class PerturbationEvaluator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def evaluate(
        self,
        *,
        predictions: dict[str, np.ndarray],
        truths: dict[str, np.ndarray],
        controls: dict[str, np.ndarray],
        top_de_indices: dict[str, tuple[int, ...]],
        strata: dict[str, dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        conditions = tuple(sorted(predictions))
        if not conditions or set(conditions) != set(truths) or set(conditions) != set(controls):
            raise ValueError("prediction, truth, and control condition sets must match and be non-empty")
        per_condition: dict[str, dict[str, float | None]] = {}
        for offset, condition in enumerate(conditions):
            per_condition[condition] = condition_metrics(
                prediction=predictions[condition],
                truth=truths[condition],
                controls=controls[condition],
                top_de_indices=top_de_indices.get(condition, ()),
                compute_energy=bool(self.config.get("energy_distance", False)),
                energy_max_samples=int(self.config.get("energy_max_samples", 64)),
                seed=int(self.config.get("bootstrap_seed", 1)) + offset,
            )
        summary = self._summarize(per_condition, conditions)
        stratified: dict[str, Any] = {}
        for stratum_name, labels in sorted((strata or {}).items()):
            if set(labels) != set(conditions):
                raise ValueError(f"stratum {stratum_name!r} does not cover the evaluated conditions")
            groups: dict[str, list[str]] = {}
            for condition in conditions:
                groups.setdefault(labels[condition], []).append(condition)
            stratified[stratum_name] = {
                label: self._summarize(per_condition, tuple(group_conditions))
                for label, group_conditions in sorted(groups.items())
            }
        return {
            "statistical_unit": "perturbation_condition",
            "conditions": list(conditions),
            "per_condition": per_condition,
            "summary": summary,
            "stratified": stratified,
        }

    def _summarize(
        self,
        per_condition: dict[str, dict[str, float | None]],
        conditions: tuple[str, ...],
    ) -> dict[str, Any]:
        metric_names = tuple(sorted({name for row in per_condition.values() for name in row}))
        summary: dict[str, Any] = {}
        bootstrap = int(self.config.get("bootstrap_resamples", 0))
        seed = int(self.config.get("bootstrap_seed", 1))
        for metric_offset, metric in enumerate(metric_names):
            rng = np.random.default_rng(seed + metric_offset)
            values = np.asarray(
                [per_condition[condition].get(metric, np.nan) for condition in conditions],
                dtype=np.float64,
            )
            finite = np.isfinite(values)
            item: dict[str, Any] = {
                "macro_mean": float(values[finite].mean()) if finite.any() else None,
                "finite_conditions": int(finite.sum()),
                "total_conditions": len(conditions),
            }
            if bootstrap > 0 and finite.sum() >= 2:
                samples = values[finite]
                means = np.asarray(
                    [rng.choice(samples, len(samples), replace=True).mean() for _ in range(bootstrap)]
                )
                item["condition_bootstrap_95ci"] = [
                    float(np.quantile(means, 0.025)),
                    float(np.quantile(means, 0.975)),
                ]
            summary[metric] = item
        return summary
