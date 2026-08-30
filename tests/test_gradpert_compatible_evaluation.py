import json
from types import SimpleNamespace

import numpy as np
import pytest

from dinogenept.evaluation.gradpert import (
    GradPertEvaluator,
    _sha256_array,
    _sha256_json,
    frozen_control_rows,
)
from dinogenept.experiments.runner import build_input_manifest
from dinogenept.priors import PriorStore
from genept_seed.provenance import digest_file


def _repeat(values, rows=300):
    return np.repeat(np.asarray([values], dtype=np.float32), rows, axis=0)


def test_gradpert_evaluator_preserves_three_distinct_reference_contracts(tmp_path):
    genes = ("A", "B", "C", "D")
    genes_path = tmp_path / "expression_gene_ids.txt"
    genes_path.write_text("\n".join(genes) + "\n", encoding="utf-8")
    arrays_path = tmp_path / "state_arrays.npz"
    systema_reference = np.asarray([2.0, 1.0, 1.0, 5.0], dtype=np.float32)
    metric_control_means = np.asarray(
        [[0.0, 2.0, 3.0, 1.0]], dtype=np.float32
    )
    np.savez_compressed(
        arrays_path,
        systema_reference=systema_reference,
        metric_control_means=metric_control_means,
    )
    state_path = tmp_path / "state_manifest.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": "evaluation-state-v1",
                "dataset_id": "fixture",
                "protocol_id": "fixture-protocol",
                "split_content_sha256": "a" * 64,
                "canonical_data_sha256": "b" * 64,
                "arrays_sha256": digest_file(arrays_path),
                "condition_ids": ["PERT_A"],
                "condition_ids_sha256": _sha256_json(["PERT_A"]),
                "expression_gene_order_sha256": _sha256_json(list(genes)),
                "de_gene_indices": {"PERT_A": [0, 1, 3]},
                "de_gene_indices_sha256": _sha256_json({"PERT_A": [0, 1, 3]}),
                "top_de_gene_indices": {"PERT_A": [1, 2, 3]},
                "top_de_gene_indices_sha256": _sha256_json(
                    {"PERT_A": [1, 2, 3]}
                ),
                "de_unavailable_reasons": {},
                "de_unavailable_reasons_sha256": _sha256_json({}),
                "de_method": (
                    "scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"
                ),
                "de_reference": "ctrl",
                "systema_reference_condition_ids": ["PERT_A"],
                "systema_reference_condition_ids_sha256": _sha256_json(["PERT_A"]),
                "systema_reference_content_sha256": _sha256_array(systema_reference),
                "metric_control_means_content_sha256": _sha256_array(
                    metric_control_means
                ),
            }
        ),
        encoding="utf-8",
    )
    evaluator = GradPertEvaluator(
        {
            "expression_gene_ids_path": str(genes_path),
            "state_manifest": str(state_path),
            "state_arrays": str(arrays_path),
            "dataset_id": "fixture",
            "data_protocol_id": "fixture-protocol",
            "split_content_sha256": "a" * 64,
            "canonical_data_sha256": "b" * 64,
            "state_condition_ids": ["PERT_A"],
        }
    )
    prediction = _repeat([3.0, 5.0, 4.0, 8.0])
    control = _repeat([1.0, 1.0, 2.0, 2.0])
    truth = _repeat([4.0, 3.0, 6.0, 7.0], rows=7)
    result = evaluator.evaluate(
        predictions={"PERT_A": prediction},
        truths={"PERT_A": truth},
        controls={"PERT_A": control},
        top_de_indices={},
        gene_ids=genes,
    )
    observed = [result["summary"][metric]["macro_mean"] for metric in result["summary"]]
    expected = [
        np.corrcoef(prediction.mean(0) - control.mean(0), truth.mean(0) - control.mean(0))[0, 1],
        np.corrcoef(
            (prediction.mean(0) - np.asarray([0.0, 2.0, 3.0, 1.0]))[[0, 1, 3]],
            (truth.mean(0) - np.asarray([0.0, 2.0, 3.0, 1.0]))[[0, 1, 3]],
        )[0, 1],
        np.corrcoef(
            (prediction.mean(0) - np.asarray([2.0, 1.0, 1.0, 5.0]))[[1, 2, 3]],
            (truth.mean(0) - np.asarray([2.0, 1.0, 1.0, 5.0]))[[1, 2, 3]],
        )[0, 1],
    ]
    assert observed == pytest.approx(expected)
    assert len({round(value, 8) for value in observed}) == 3


def test_frozen_control_rows_preserve_order_duplicates_and_seed(tmp_path):
    row_ids = ("ctrl-0", "ctrl-1", "pert-0")
    expression = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    selected = ["ctrl-1", "ctrl-0"] * 150
    contexts = ["context-1", "context-0"] * 150
    manifest_path = tmp_path / "evaluation_controls.test.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "evaluation-controls-v1",
                "dataset_id": "fixture",
                "protocol_id": "fixture-protocol",
                "split_content_sha256": "a" * 64,
                "split_name": "test",
                "evaluation_seed": 20260824,
                "rng": "numpy_pcg64",
                "sample_with_replacement": True,
                "context_policy": "truth_cell_context_resampling",
                "n_controls_per_condition": 300,
                "draws": [
                    {
                        "condition_id": "PERT",
                        "context_policy": "truth_cell_context_resampling",
                        "source_pool_sha256": "b" * 64,
                        "ordered_context_ids": contexts,
                        "ordered_context_ids_sha256": _sha256_json(contexts),
                        "ordered_row_ids": selected,
                        "ordered_row_ids_sha256": _sha256_json(selected),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    data = SimpleNamespace(
        row_ids=row_ids,
        name="fixture",
        protocol_id="fixture-protocol",
        split_content_sha256="a" * 64,
        expression=expression,
        control_mask=np.asarray([True, True, False]),
        split_conditions=lambda split: ("PERT",) if split == "test" else (),
    )
    controls, observed_ids, receipt = frozen_control_rows(
        data, {"control_manifest": str(manifest_path)}, "test"
    )
    assert observed_ids["PERT"] == tuple(selected)
    assert controls["PERT"].shape == (300, 2)
    np.testing.assert_array_equal(controls["PERT"][:2], expression[[1, 0]])
    assert receipt["evaluation_seed"] == 20260824


def test_run_input_manifest_binds_every_frozen_evaluation_artifact(tmp_path):
    paths = {}
    for name in (
        "evaluation_controls.test.json",
        "state_manifest.json",
        "state_arrays.npz",
        "expression_gene_ids.txt",
    ):
        path = tmp_path / name
        path.write_bytes(f"artifact:{name}".encode())
        paths[name] = path
    config = {
        "model": {"pretrained_checkpoint": None},
        "evaluation": {
            "name": "gradpert_exact",
            "protocol_id": "gradpert-metrics-v1",
            "split": "test",
            "seed": 20260824,
            "control_samples": 300,
            "control_manifest": str(paths["evaluation_controls.test.json"]),
            "state_manifest": str(paths["state_manifest.json"]),
            "state_arrays": str(paths["state_arrays.npz"]),
            "expression_gene_ids_path": str(paths["expression_gene_ids.txt"]),
        },
    }
    data = SimpleNamespace(
        fingerprint="f" * 64,
        source_sha256="s" * 64,
        split_sha256="p" * 64,
        protocol_id="fixture-protocol",
        split_content_sha256="c" * 64,
        expression_gene_order_sha256="g" * 64,
    )
    priors = PriorStore.from_config(
        {
            "priors": {
                "base": {
                    "fixture": "deterministic",
                    "dimension": 8,
                    "coverage": 1.0,
                },
                "optional": {},
            }
        },
        ("A",),
    )
    manifest = build_input_manifest(config, data, priors)
    artifacts = manifest["evaluation"]["artifacts"]
    assert set(artifacts) == {
        "control_manifest",
        "state_manifest",
        "state_arrays",
        "expression_gene_ids_path",
    }
    assert all(len(item["sha256"]) == 64 for item in artifacts.values())
