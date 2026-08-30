import numpy as np

from dinogenept.datasets.common import (
    _column_variance,
    _extract_top_de,
    load_perturbation_dataset,
    plus_condition_parser,
)


def test_column_variance_matches_numpy():
    matrix = np.arange(40, dtype=np.float32).reshape(8, 5)
    rows = np.asarray([0, 2, 4, 7])
    observed = _column_variance(matrix, rows, chunk_rows=2)
    assert np.allclose(observed, np.var(matrix[rows], axis=0))


def test_npz_loader_preserves_rank_two_after_row_and_column_selection(tmp_path):
    genes = np.asarray(["A", "B", "C", "D"])
    conditions = np.asarray(
        ["ctrl", "ctrl", "A+ctrl", "B+ctrl", "C+ctrl", "D+ctrl", "E+ctrl"]
    )
    expression = np.arange(28, dtype=np.float32).reshape(7, 4)
    source = tmp_path / "fixture.npz"
    np.savez(source, expression=expression, genes=genes, conditions=conditions)
    config = {
        "dataset": {
            "path": str(source),
            "control_label": "ctrl",
            "split": {
                "strategy": "deterministic_condition",
                "seed": 7,
                "test_fraction": 0.2,
                "validation_fraction": 0.2,
            },
            "limits": {
                "max_conditions": 0,
                "max_cells_per_condition": 0,
                "max_control_cells": 0,
                "max_genes": 2,
                "variance_chunk_rows": 2,
            },
        }
    }
    loaded = load_perturbation_dataset(
        config, name="fixture", parse_condition=plus_condition_parser
    )
    assert loaded.expression.shape == (7, 2)
    assert loaded.n_genes == 2
    assert len(loaded.source_sha256) == 64
    assert len(loaded.split_sha256) == 64
    assert loaded.ibot_gene_indices
    assert set(loaded.split_by_condition.values()) == {"train", "validation", "test"}


def test_top_de_integer_indices_are_remapped_to_selected_gene_axis():
    observed = _extract_top_de(
        {"de": {"A+ctrl": [4, 1, 3]}},
        "de",
        ("B", "D", "E"),
        np.asarray([1, 3, 4]),
    )
    assert observed == {"A+ctrl": (2, 0, 1)}
