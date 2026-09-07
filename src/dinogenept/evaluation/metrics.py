"""GraD-Pert-aligned metrics, independently implemented from frozen definitions.

Reference: GraD-Pert 276d7baac57c6a4130b5e43563cc7eccd8c8ff7e,
evaluation/metrics.py. Test-only parity may inspect that checkout; production
does not import it. Evaluation truth and reference populations are never model
features. Row identities and frozen DE provenance are the caller's contract.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

METRICS = ("txpert_macro_pearson_delta", "trishift_pearson_delta", "systema_pearson")


@dataclass(frozen=True)
class Score:
    value: float | None
    reason: str | None
    gene_count: int


def pearson(left: np.ndarray, right: np.ndarray) -> Score:
    x, y = (np.asarray(a, dtype=np.float64) for a in (left, right))
    if x.ndim != 1 or x.shape != y.shape or not x.size:
        raise ValueError("Pearson requires equal nonempty vectors")
    if x.size < 2:
        return Score(None, "fewer_than_two_genes", x.size)
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return Score(None, "non_finite_input", x.size)
    dx, dy = x - x.mean(), y - y.mean()
    denominator = float(np.sqrt(np.dot(dx, dx) * np.dot(dy, dy)))
    if denominator == 0:
        return Score(None, "constant_vector", x.size)
    return Score(float(np.dot(dx, dy) / denominator), None, x.size)


def _population(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError("Expected a nonempty cell-by-gene matrix")
    return array


def _index(values: Sequence[int], width: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("DE indices must be nonempty integer vectors")
    if len(np.unique(array)) != array.size or array.min() < 0 or array.max() >= width:
        raise ValueError("DE indices duplicate or out of range")
    return array.astype(np.int64)


def evaluate_condition(
    prediction: np.ndarray,
    input_control: np.ndarray,
    truth: np.ndarray,
    metric_control_mean: np.ndarray,
    reference: np.ndarray,
    de_indices: Sequence[int],
    top_de_indices: Sequence[int],
    *,
    de_unavailable_reason: str | None = None,
) -> dict[str, Score]:
    """Evaluate one condition on 300 input controls and all its truth cells."""
    pred, ctrl, real = map(_population, (prediction, input_control, truth))
    if pred.shape != ctrl.shape or pred.shape[0] != 300 or real.shape[1] != pred.shape[1]:
        raise ValueError("Pred/InputCtrl must have matching [300,G] shapes; Truth must share G")
    width = pred.shape[1]
    pool, ref = (np.asarray(a, dtype=np.float64) for a in (metric_control_mean, reference))
    if pool.shape != (width,) or ref.shape != (width,):
        raise ValueError("Reference gene axes differ")
    pm, tm, cm = pred.mean(0), real.mean(0), ctrl.mean(0)
    scores = {METRICS[0]: pearson(pm - cm, tm - cm)}
    if de_unavailable_reason is not None:
        if not isinstance(de_unavailable_reason, str) or not de_unavailable_reason:
            raise ValueError("Unavailable DE needs a nonempty reason")
        if len(de_indices) or len(top_de_indices):
            raise ValueError("Unavailable DE must have empty indices")
        scores.update({key: Score(None, f"de_unavailable:{de_unavailable_reason}", 0) for key in METRICS[1:]})
    else:
        de, top = _index(de_indices, width), _index(top_de_indices, width)
        scores[METRICS[1]] = pearson((pm - pool)[de], (tm - pool)[de])
        scores[METRICS[2]] = pearson((pm - ref)[top], (tm - ref)[top])
    return scores


def systema_reference(populations: Mapping[str, np.ndarray]) -> np.ndarray:
    """Equal weight per noncontrol condition, regardless of cell count."""
    if not populations or "ctrl" in populations:
        raise ValueError("Systema requires noncontrol train+validation populations")
    means = [_population(populations[key]).mean(0) for key in sorted(populations)]
    return np.stack(means).mean(0)


def macro_average(conditions: Mapping[str, Mapping[str, Score]]) -> dict:
    """Retain the denominator and missingness; never replace undefined with zero."""
    if not conditions or any(set(row) != set(METRICS) for row in conditions.values()):
        raise ValueError("Each condition must contain exactly the three metrics")
    result = {}
    for key in METRICS:
        rows = [row[key] for row in conditions.values()]
        for score in rows:
            if (score.value is None) != (score.reason is not None):
                raise ValueError("Metric value/reason inconsistent")
            if score.value is not None and not np.isfinite(score.value):
                raise ValueError("Nonfinite scores must be represented as unavailable")
        valid = [row.value for row in rows if row.value is not None]
        result[key] = {
            "macro_mean": float(np.mean(valid)) if valid else None,
            "finite_condition_count": len(valid),
            "total_condition_count": len(rows),
            "unavailable_reasons": sorted(row.reason for row in rows if row.reason is not None),
        }
    return {"metrics": result, "conditions": {c: {m: asdict(s) for m, s in r.items()} for c, r in conditions.items()}}
