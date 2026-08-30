"""GraD-Pert-compatible frozen-control evaluation without a runtime dependency."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from genept_seed.provenance import digest_file

METRIC_IDS = (
    "txpert_macro_pearson_delta",
    "trishift_pearson_delta",
    "systema_pearson",
)


def _read_json(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(str(path)).resolve(strict=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON contract must be a mapping: {source}")
    return source, payload


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(_sha256_json(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validate_hash_field(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"GraD-Pert {field} is not a lowercase SHA-256")


def frozen_control_rows(
    data: Any, evaluation: dict[str, Any], split: str
) -> tuple[dict[str, np.ndarray], dict[str, tuple[str, ...]], dict[str, Any]]:
    """Load the exact ordered 300-row draws shared by all compared models."""

    path, manifest = _read_json(evaluation["control_manifest"])
    expected_split = "val" if split == "validation" else split
    if (
        manifest.get("schema_version") != "evaluation-controls-v1"
        or manifest.get("dataset_id") != data.name
        or manifest.get("protocol_id") != data.protocol_id
        or manifest.get("split_content_sha256") != data.split_content_sha256
        or manifest.get("split_name") != expected_split
        or int(manifest.get("evaluation_seed", -1)) != 20260824
        or manifest.get("rng") != "numpy_pcg64"
        or manifest.get("sample_with_replacement") is not True
        or manifest.get("context_policy") != "truth_cell_context_resampling"
        or int(manifest.get("n_controls_per_condition", -1)) != 300
    ):
        raise ValueError("frozen GraD-Pert control manifest contract differs")
    expected_conditions = data.split_conditions(split)
    draws = manifest.get("draws")
    if not isinstance(draws, list):
        raise ValueError("frozen control manifest draws must be a list")
    by_condition = {str(item["condition_id"]): item for item in draws}
    if len(by_condition) != len(draws):
        raise ValueError("frozen control manifest has duplicate conditions")
    if set(by_condition) != set(expected_conditions):
        raise ValueError("frozen control draws differ from the requested split")
    row_index = {row_id: index for index, row_id in enumerate(data.row_ids)}
    controls: dict[str, np.ndarray] = {}
    row_ids: dict[str, tuple[str, ...]] = {}
    for condition in expected_conditions:
        draw = by_condition[condition]
        selected = tuple(str(value) for value in draw["ordered_row_ids"])
        contexts = tuple(str(value) for value in draw.get("ordered_context_ids", ()))
        if len(selected) != 300:
            raise ValueError(f"frozen control draw must contain 300 rows: {condition}")
        if (
            len(contexts) != 300
            or draw.get("context_policy") != "truth_cell_context_resampling"
            or _sha256_json(list(selected)) != draw.get("ordered_row_ids_sha256")
            or _sha256_json(list(contexts)) != draw.get("ordered_context_ids_sha256")
        ):
            raise ValueError(f"frozen control draw content hash differs: {condition}")
        _validate_hash_field(draw.get("source_pool_sha256"), "source_pool_sha256")
        try:
            indices = [row_index[row_id] for row_id in selected]
        except KeyError as error:
            raise ValueError(f"frozen control row is absent from loaded data: {error.args[0]}") from error
        if not bool(data.control_mask[np.asarray(indices, dtype=np.int64)].all()):
            raise ValueError("frozen control draw contains a perturbed row")
        controls[condition] = np.ascontiguousarray(data.expression[indices], dtype=np.float32)
        row_ids[condition] = selected
    return controls, row_ids, {
        "path": str(path),
        "sha256": digest_file(path),
        "evaluation_seed": 20260824,
        "rows_per_condition": 300,
        "dataset_id": data.name,
        "protocol_id": data.protocol_id,
        "split_content_sha256": data.split_content_sha256,
    }


def _pearson(left: np.ndarray, right: np.ndarray) -> tuple[float | None, str | None]:
    x = np.asarray(left, dtype=np.float64).reshape(-1)
    y = np.asarray(right, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError("Pearson vectors have different shapes")
    if x.size < 2:
        return None, "fewer_than_two_genes"
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, "non_finite_input"
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator == 0:
        return None, "constant_vector"
    return float(np.dot(x, y) / denominator), None


class GradPertEvaluator:
    """Reproduce the three frozen GraD-Pert condition-macro metrics."""

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
        gene_ids: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        del top_de_indices
        conditions = tuple(sorted(predictions))
        if not conditions or set(conditions) != set(truths) or set(conditions) != set(controls):
            raise ValueError("prediction, truth, and control condition sets must match")
        if gene_ids is None:
            raise ValueError("GraD-Pert evaluation requires the frozen gene order")
        gene_path = Path(str(self.config["expression_gene_ids_path"])).resolve(strict=True)
        expected_genes = tuple(gene_path.read_text(encoding="utf-8").splitlines())
        if gene_ids != expected_genes:
            raise ValueError("model output gene order differs from GraD-Pert expression genes")

        state_path, manifest = _read_json(self.config["state_manifest"])
        arrays_path = Path(str(self.config["state_arrays"])).resolve(strict=True)
        if manifest.get("schema_version") != "evaluation-state-v1":
            raise ValueError("unsupported GraD-Pert evaluation-state schema")
        if (
            manifest.get("dataset_id") != self.config["dataset_id"]
            or manifest.get("protocol_id") != self.config["data_protocol_id"]
            or manifest.get("split_content_sha256") != self.config["split_content_sha256"]
            or manifest.get("canonical_data_sha256")
            != self.config["canonical_data_sha256"]
            or manifest.get("expression_gene_order_sha256")
            != _sha256_json(list(gene_ids))
        ):
            raise ValueError("GraD-Pert evaluation-state identity differs")
        if digest_file(arrays_path) != manifest.get("arrays_sha256"):
            raise ValueError("GraD-Pert evaluation-state arrays hash differs")
        condition_order = tuple(str(item) for item in manifest["condition_ids"])
        if _sha256_json(list(condition_order)) != manifest.get("condition_ids_sha256"):
            raise ValueError("GraD-Pert evaluation-state condition hash differs")
        if condition_order != tuple(self.config["state_condition_ids"]):
            raise ValueError("GraD-Pert evaluation-state conditions differ from frozen split")
        condition_index = {condition: index for index, condition in enumerate(condition_order)}
        if any(condition not in condition_index for condition in conditions):
            raise ValueError("evaluation state lacks a requested test condition")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"systema_reference", "metric_control_means"}:
                raise ValueError("GraD-Pert evaluation-state array keys differ")
            systema_reference = np.asarray(arrays["systema_reference"], dtype=np.float64)
            metric_controls = np.asarray(arrays["metric_control_means"], dtype=np.float64)
            raw_systema_reference = np.asarray(arrays["systema_reference"])
            raw_metric_controls = np.asarray(arrays["metric_control_means"])
        if systema_reference.shape != (len(gene_ids),) or metric_controls.shape != (
            len(condition_order),
            len(gene_ids),
        ):
            raise ValueError("GraD-Pert evaluation-state array dimensions differ")
        if (
            _sha256_array(raw_systema_reference)
            != manifest.get("systema_reference_content_sha256")
            or _sha256_array(raw_metric_controls)
            != manifest.get("metric_control_means_content_sha256")
            or not np.isfinite(systema_reference).all()
            or not np.isfinite(metric_controls).all()
        ):
            raise ValueError("GraD-Pert evaluation-state array content differs")

        de_by_condition = manifest["de_gene_indices"]
        top_de_by_condition = manifest["top_de_gene_indices"]
        unavailable = manifest.get("de_unavailable_reasons", {})
        reference_conditions = manifest.get("systema_reference_condition_ids")
        if (
            not isinstance(reference_conditions, list)
            or _sha256_json(reference_conditions)
            != manifest.get("systema_reference_condition_ids_sha256")
        ):
            raise ValueError("GraD-Pert Systema reference-condition hash differs")
        if (
            manifest.get("de_method")
            != "scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"
            or manifest.get("de_reference") != "ctrl"
        ):
            raise ValueError("GraD-Pert DE method contract differs")
        for values, field in (
            (de_by_condition, "de_gene_indices"),
            (top_de_by_condition, "top_de_gene_indices"),
        ):
            if not isinstance(values, dict) or _sha256_json(values) != manifest.get(
                f"{field}_sha256"
            ):
                raise ValueError(f"GraD-Pert {field} content hash differs")
            for condition, indices in values.items():
                if condition not in condition_index or len(indices) != len(set(indices)):
                    raise ValueError(f"GraD-Pert {field} condition or uniqueness differs")
                if any(not isinstance(index, int) or not 0 <= index < len(gene_ids) for index in indices):
                    raise ValueError(f"GraD-Pert {field} index is outside gene axis")
        if _sha256_json(unavailable) != manifest.get("de_unavailable_reasons_sha256"):
            raise ValueError("GraD-Pert DE-unavailable content hash differs")
        per_condition: dict[str, dict[str, dict[str, Any]]] = {}
        for condition in conditions:
            prediction = np.asarray(predictions[condition], dtype=np.float64)
            truth = np.asarray(truths[condition], dtype=np.float64)
            control = np.asarray(controls[condition], dtype=np.float64)
            if prediction.shape != (300, len(gene_ids)) or control.shape != prediction.shape:
                raise ValueError("GraD-Pert prediction and control must preserve [300, genes]")
            if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] != len(gene_ids):
                raise ValueError("GraD-Pert truth population has an invalid shape")
            prediction_mean = prediction.mean(axis=0)
            truth_mean = truth.mean(axis=0)
            input_control_mean = control.mean(axis=0)
            metric_control = metric_controls[condition_index[condition]]
            reason = unavailable.get(condition)
            values: dict[str, tuple[float | None, str | None, int]] = {}
            tx_value, tx_reason = _pearson(
                prediction_mean - input_control_mean,
                truth_mean - input_control_mean,
            )
            values[METRIC_IDS[0]] = (tx_value, tx_reason, len(gene_ids))
            if reason is None:
                de = np.asarray(de_by_condition[condition], dtype=np.int64)
                top_de = np.asarray(top_de_by_condition[condition], dtype=np.int64)
                tri_value, tri_reason = _pearson(
                    (prediction_mean - metric_control)[de],
                    (truth_mean - metric_control)[de],
                )
                sys_value, sys_reason = _pearson(
                    (prediction_mean - systema_reference)[top_de],
                    (truth_mean - systema_reference)[top_de],
                )
                values[METRIC_IDS[1]] = (tri_value, tri_reason, int(de.size))
                values[METRIC_IDS[2]] = (sys_value, sys_reason, int(top_de.size))
            else:
                unavailable_reason = f"de_unavailable:{reason}"
                values[METRIC_IDS[1]] = (None, unavailable_reason, 0)
                values[METRIC_IDS[2]] = (None, unavailable_reason, 0)
            per_condition[condition] = {
                metric_id: {"value": value, "reason": why, "gene_count": count}
                for metric_id, (value, why, count) in values.items()
            }

        summary: dict[str, Any] = {}
        for metric_id in METRIC_IDS:
            values = [per_condition[c][metric_id]["value"] for c in conditions]
            finite = [float(value) for value in values if value is not None]
            summary[metric_id] = {
                "macro_mean": float(np.mean(finite)) if finite else None,
                "finite_conditions": len(finite),
                "total_conditions": len(conditions),
                "unavailable_reasons": sorted(
                    row[metric_id]["reason"]
                    for row in per_condition.values()
                    if row[metric_id]["reason"] is not None
                ),
            }
        stratified: dict[str, Any] = {}
        for name, labels in sorted((strata or {}).items()):
            if set(labels) != set(conditions):
                raise ValueError(f"stratum {name!r} does not cover evaluated conditions")
            groups: dict[str, list[str]] = {}
            for condition in conditions:
                groups.setdefault(labels[condition], []).append(condition)
            stratified[name] = {
                label: {
                    metric_id: float(
                        np.mean(
                            [
                                per_condition[condition][metric_id]["value"]
                                for condition in members
                                if per_condition[condition][metric_id]["value"] is not None
                            ]
                        )
                    )
                    if any(
                        per_condition[condition][metric_id]["value"] is not None
                        for condition in members
                    )
                    else None
                    for metric_id in METRIC_IDS
                }
                for label, members in sorted(groups.items())
            }
        return {
            "schema_version": "gradpert-compatible-evaluation-v1",
            "metric_registry": "gradpert-metrics-v1",
            "statistical_unit": "perturbation_condition",
            "conditions": list(conditions),
            "per_condition": per_condition,
            "summary": summary,
            "stratified": stratified,
            "provenance": {
                "state_manifest": str(state_path),
                "state_manifest_sha256": digest_file(state_path),
                "state_arrays": str(arrays_path),
                "state_arrays_sha256": digest_file(arrays_path),
                "expression_gene_ids": str(gene_path),
                "expression_gene_ids_sha256": digest_file(gene_path),
            },
        }


def build_evaluator(config: dict[str, Any]) -> GradPertEvaluator:
    return GradPertEvaluator(config["evaluation"])
