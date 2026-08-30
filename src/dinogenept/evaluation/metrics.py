"""Condition-level metrics shared by every perturbation model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _population(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a non-empty rank-two array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return None if denominator == 0 else float(np.dot(x, y) / denominator)


def energy_distance(
    prediction: np.ndarray, truth: np.ndarray, *, max_samples: int, seed: int
) -> float:
    predicted = _population(prediction, "prediction")
    observed = _population(truth, "truth")
    rng = np.random.default_rng(seed)
    if len(predicted) > max_samples:
        predicted = predicted[rng.choice(len(predicted), max_samples, replace=False)]
    if len(observed) > max_samples:
        observed = observed[rng.choice(len(observed), max_samples, replace=False)]

    def mean_distance(left: np.ndarray, right: np.ndarray) -> float:
        squared = (
            np.sum(left * left, axis=1)[:, None]
            + np.sum(right * right, axis=1)[None, :]
            - 2.0 * left @ right.T
        )
        return float(np.sqrt(np.maximum(squared, 0.0)).mean())

    value = 2.0 * mean_distance(predicted, observed)
    value -= mean_distance(predicted, predicted)
    value -= mean_distance(observed, observed)
    return float(max(value, 0.0))


def condition_metrics(
    *,
    prediction: np.ndarray,
    truth: np.ndarray,
    controls: np.ndarray,
    top_de_indices: tuple[int, ...] = (),
    compute_energy: bool = False,
    energy_max_samples: int = 64,
    seed: int = 1,
) -> dict[str, float | None]:
    predicted = _population(prediction, "prediction")
    observed = _population(truth, "truth")
    control = _population(controls, "controls")
    if predicted.shape[1] != observed.shape[1] or predicted.shape[1] != control.shape[1]:
        raise ValueError("prediction, truth, and controls have different gene axes")
    pred_mean = predicted.mean(axis=0)
    truth_mean = observed.mean(axis=0)
    ctrl_mean = control.mean(axis=0)
    pred_delta = pred_mean - ctrl_mean
    truth_delta = truth_mean - ctrl_mean
    mse = float(np.mean((pred_mean - truth_mean) ** 2))
    denominator = float(np.mean((ctrl_mean - truth_mean) ** 2))
    indices = np.asarray(top_de_indices, dtype=np.int64)
    if indices.size and (indices.min() < 0 or indices.max() >= pred_mean.size):
        raise ValueError("top-DE indices fall outside the gene axis")
    result: dict[str, float | None] = {
        "mse": mse,
        "mse_delta": float(np.mean((pred_delta - truth_delta) ** 2)),
        "normalized_mse": None if denominator == 0 else mse / denominator,
        "pearson_delta": pearson(pred_delta, truth_delta),
        "pearson_expression": pearson(pred_mean, truth_mean),
        "direction_accuracy": float(np.mean(np.sign(pred_delta) == np.sign(truth_delta))),
        "top20_pearson_delta": (
            pearson(pred_delta[indices], truth_delta[indices]) if indices.size else None
        ),
        "top20_mse_delta": (
            float(np.mean((pred_delta[indices] - truth_delta[indices]) ** 2))
            if indices.size
            else None
        ),
        "top20_direction_accuracy": (
            float(
                np.mean(
                    np.sign(pred_delta[indices]) == np.sign(truth_delta[indices])
                )
            )
            if indices.size
            else None
        ),
    }
    if compute_energy:
        result["energy_distance"] = energy_distance(
            predicted, observed, max_samples=energy_max_samples, seed=seed
        )
    return result
