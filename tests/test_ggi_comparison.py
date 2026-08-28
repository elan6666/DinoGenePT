import json

import pytest

from genept_seed.ggi_comparison import FAIRNESS_FIELDS, build_ggi_comparison


def _write_result(path, name, accuracy, *, random_state=42):
    fairness = {
        "task": "gene_gene_interaction",
        "model": "logistic_regression",
        "normalization": "l2",
        "pair_operator": "sum",
        "random_state": random_state,
        "train_n": 10,
        "test_n": 4,
        "train_coverage": 1.0,
        "test_coverage": 1.0,
        "data_receipt_sha256": "data",
        "gene_universe": "genes.txt",
        "gene_universe_sha256": "genes",
        "genept_seed_version": "0.1.0",
        "numpy_version": "1",
        "scikit_learn_version": "1",
    }
    assert set(fairness) == set(FAIRNESS_FIELDS)
    path.write_text(
        json.dumps(
            [
                {
                    "embedding": name,
                    "vectors_sha256": name,
                    "accuracy": accuracy,
                    "auroc": 0.8,
                    "average_precision": 0.7,
                    **fairness,
                }
            ]
        )
    )


def test_build_ggi_comparison_enforces_fairness_and_computes_deltas(tmp_path):
    baseline, enriched = tmp_path / "base.json", tmp_path / "enriched.json"
    _write_result(baseline, "base", 0.7)
    _write_result(enriched, "enriched", 0.72)
    receipt = build_ggi_comparison(
        result_paths=[baseline, enriched],
        baseline="base",
        output_path=tmp_path / "comparison.json",
    )
    assert receipt["fairness_fields_identical"] is True
    assert receipt["conditions"][1]["delta_vs_baseline"]["accuracy"] == pytest.approx(0.02)


def test_build_ggi_comparison_rejects_protocol_drift(tmp_path):
    baseline, drift = tmp_path / "base.json", tmp_path / "drift.json"
    _write_result(baseline, "base", 0.7)
    _write_result(drift, "drift", 0.7, random_state=7)
    with pytest.raises(ValueError, match="fairness fields differ"):
        build_ggi_comparison(
            result_paths=[baseline, drift],
            baseline="base",
            output_path=tmp_path / "comparison.json",
        )
