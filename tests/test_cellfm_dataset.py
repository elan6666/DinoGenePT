import json
from dataclasses import replace

import numpy as np
import pytest
from scipy import sparse

from dinogenept.datasets.cellfm.dataset import CellFMDataset
from dinogenept.datasets.cellfm.mapping import map_axis
from dinogenept.provenance import atomic_write_json, digest_file

ad = pytest.importorskip("anndata")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")


@pytest.fixture
def frozen(tmp_path):
    directory = tmp_path / "frozen"
    directory.mkdir()
    conditions = ["ctrl"] * 4 + ["A+ctrl"] * 5 + ["B+ctrl"] * 3 + ["C+ctrl"] * 2
    obs = pd.DataFrame(
        {"condition": conditions, "cell_type": ["K562"] * 14, "condition_name": conditions},
        index=[f"row-{i}" for i in range(14)],
    )
    axis = ["C", "A", "B"]
    values = np.arange(42, dtype=np.float32).reshape(14, 3) / 7
    values[:, 0] = 0  # All-zero measured genes must not disappear in stage 2.
    source = tmp_path / "official.h5ad"
    data = ad.AnnData(sparse.csr_matrix(values), obs=obs, var=pd.DataFrame({"gene_name": axis}, index=axis))
    data.uns["must_not_read"] = {"heldout_DE": ["C", "B", "A"]}
    data.write_h5ad(source)
    observations = obs.copy()
    observations.insert(0, "source_row", np.arange(14))
    observations.insert(0, "row_id", obs.index)
    observations.to_parquet(directory / "observations.parquet", index=False)
    atomic_write_json(directory / "axis_symbols.json", axis)
    atomic_write_json(directory / "target_symbols.json", ["A", "B", "C"])
    atomic_write_json(
        directory / "split.json", {"conditions": {"train": ["ctrl", "A+ctrl"], "val": ["B+ctrl"], "test": ["C+ctrl"]}}
    )
    audit = {
        "source": str(source),
        "source_sha256": digest_file(source),
        "cells": 14,
        "genes": 3,
        "expression": "official_provided_log1p_matrix_no_renormalization",
        "files_sha256": {path.name: digest_file(path) for path in directory.iterdir()},
    }
    atomic_write_json(directory / "audit.json", audit)
    vocab = tmp_path / "genes.json"
    atomic_write_json(vocab, ["EA", "EB", "EC"])
    mapping = map_axis(
        axis, ["A", "B", "C"], ["EA", "EB", "EC"], [{"feature_name": gene, "feature_id": f"E{gene}"} for gene in "ABC"]
    )
    mapping.update(vocabulary_sha256=digest_file(vocab), downstream_audit_sha256=digest_file(directory / "audit.json"))
    atomic_write_json(directory / "gene_mapping.json", mapping)
    return {
        "directory": directory,
        "vocabulary": vocab,
        "audit_sha256": digest_file(directory / "audit.json"),
        "mapping_sha256": digest_file(directory / "gene_mapping.json"),
    }, values


def test_real_format_sparse_loader_and_bags_preserve_values_and_zeros(frozen):
    config, values = frozen
    data = CellFMDataset(**config)
    assert sparse.issparse(data._matrix)
    assert data.axis.tolist() == [3, 1, 2]
    assert not hasattr(data, "uns")
    bags = data.index.training_bags(epoch=0)
    assert sorted(i for bag in bags for i in bag.primary_post) == list(range(4, 9))
    batch = data.training_inputs(bags[0], epoch=0)
    np.testing.assert_array_equal(batch["train_post"], values[list(bags[0].primary_post)])
    np.testing.assert_array_equal(batch["full_control"], values[list(bags[0].controls)])
    np.testing.assert_array_equal(batch["control_view"]["expression"], batch["full_control"][:, [1, 2, 0]])
    assert batch["control_view"]["valid"].all()
    assert not batch["control_view"]["hidden"].any()
    assert not batch["observed_view"]["hidden"].any()
    assert not set(batch["observed_cell_ids"]) & set(batch["teacher_cell_ids"])
    assert not batch["teacher_view"]["hidden"].any()
    assert batch["targets"].tolist() == [1]


def test_heldout_cannot_become_teacher_or_observed_or_controls(frozen):
    config, _ = frozen
    data = CellFMDataset(**config)
    bag = data.index.training_bags(epoch=0)[0]
    for field in ("primary_post", "teacher_post", "observed_post", "controls"):
        with pytest.raises(ValueError, match="condition/context"):
            data.training_inputs(replace(bag, **{field: (9,)}), epoch=0)
    with pytest.raises(ValueError, match="must be in train"):
        data.training_inputs(replace(bag, condition="B+ctrl"), epoch=0)
    with pytest.raises(ValueError, match="disjoint"):
        data.training_inputs(replace(bag, observed_post=bag.teacher_post), epoch=0)


def test_optional_mask_selection_and_donor_disclosure(frozen):
    config, _ = frozen
    data = CellFMDataset(**config)
    assert data.identity["donor_guaranteed"] is False
    bag = data.index.training_bags(epoch=0)[0]
    batch = data.training_inputs(bag, epoch=0, observed_ibot=True)
    hidden = batch["observed_view"]["hidden"]
    assert hidden.any(-1).sum() in (1, 2)
    assert (hidden.sum(-1) <= 1).all()  # Three-gene fixture: minimum one target.
    twin = data.training_inputs(bag, epoch=0, observed_ibot=True)
    np.testing.assert_array_equal(hidden, twin["observed_view"]["hidden"])


def test_prediction_inputs_do_not_contain_truth_and_preserve_repeated_controls(frozen):
    config, values = frozen
    data = CellFMDataset(**config)
    group = data.index.evaluation_groups("test")[0]
    inputs = data.prediction_inputs(group["condition"], group["context"], group["control_rows"])
    assert set(inputs) == {"control_view", "full_control", "axis", "targets"}
    np.testing.assert_array_equal(inputs["full_control"], values[list(group["control_rows"])])
    np.testing.assert_array_equal(data.evaluation_truth("test", "C+ctrl", "K562"), values[12:])
    with pytest.raises(ValueError, match="held-out split"):
        data.evaluation_truth("val", "C+ctrl", "K562")
    with pytest.raises(ValueError, match="condition/context"):
        data.prediction_inputs("C+ctrl", "K562", group["truth_rows"])


@pytest.mark.parametrize("name", ["split.json", "observations.parquet", "gene_mapping.json"])
def test_mutation_of_frozen_artifacts_fails(frozen, name):
    config, _ = frozen
    path = config["directory"] / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        CellFMDataset(**config)


def test_modified_vocabulary_and_semantically_incorrect_mapping_fail(frozen):
    config, _ = frozen
    mapping_path = config["directory"] / "gene_mapping.json"
    mapping = json.loads(mapping_path.read_text())
    mapping["model_gene_ids"] = [1, 3, 2]
    atomic_write_json(mapping_path, mapping)
    config["mapping_sha256"] = digest_file(mapping_path)
    with pytest.raises(ValueError, match="Ensembl identity"):
        CellFMDataset(**config)
    config["vocabulary"].write_text('["new"]')
    with pytest.raises(ValueError, match="vocabulary differs"):
        CellFMDataset(**config)
