import json

import numpy as np
import pytest
from scipy import sparse

from dinogenept.cell.dataset import PretrainingDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.cell.sparse import validate_measured_coordinates, write_csr_shard
from dinogenept.provenance import digest_file


def corpus(root):
    vocabulary = root / "genes.json"
    vocabulary.write_text(json.dumps(["A", "B", "C"]))
    shards = []
    for i, split in enumerate(("train", "train", "validation")):
        matrix = sparse.csr_matrix(np.array([[1, 0, 2], [0, 3, 4]], dtype=np.float32))
        shards.append(write_csr_shard(root, f"shard-{i}", matrix, np.arange(2) + i * 2, [3, 7], split=split))
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "dinogenept.pretraining.csr.v1",
                "status": "training_ready",
                "purpose": "unit_fixture",
                "leakage_audit": {"status": "passed", "scope": "synthetic_fixture_only"},
                "expression": "raw_counts",
                "data_id": "fixture",
                "vocabulary": {"genes": 3, "path": "genes.json", "sha256": digest_file(vocabulary)},
                "shards": shards,
            }
        )
    )
    return path


def test_sparse_memory_maps_preserve_sampling_and_survive_cache_eviction(tmp_path):
    path = corpus(tmp_path)
    dataset = PretrainingDataset(path, "train", CropConfig(cap=3), cache_shards=1)
    dataset.verify()
    first = dataset[0]
    assert isinstance(dataset.cache[0]["data"], np.memmap)
    assert not dataset.cache[0]["data"].flags.writeable
    dataset[2]
    assert 0 not in dataset.cache
    again = dataset[0]
    for a, b in zip(first["views"], again["views"], strict=True):
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
    assert len(dataset.__getstate__()["cache"]) == 0
    assert dataset.validated == {0, 1}


def test_sparse_checksum_and_non_overwrite_guards(tmp_path):
    path = corpus(tmp_path)
    with pytest.raises(FileExistsError):
        write_csr_shard(tmp_path, "shard-0", sparse.eye(2), [10, 11], [1, 1], split="train")
    with (tmp_path / "shard-0/data.npy").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="checksum changed"):
        PretrainingDataset(path, "train", CropConfig()).verify()


def test_sparse_raw_contract_and_scope(tmp_path):
    with pytest.raises(ValueError, match="Fractional"):
        write_csr_shard(tmp_path, "bad", sparse.csr_matrix([[0.5, 0]]), [1], [1], split="train")
    with pytest.raises(ValueError, match="child"):
        write_csr_shard(tmp_path, "..", sparse.eye(2), [1, 2], [1, 1], split="train")


@pytest.mark.parametrize("constructor", [sparse.csr_array, sparse.csr_matrix])
def test_presence_supports_sparse_array_and_rejects_unmeasured_counts(constructor):
    matrix = constructor([[1, 0, 0], [0, 2, 0]])
    presence = constructor([[1, 0, 1], [0, 1, 1]])
    validate_measured_coordinates(matrix, presence, ["A", "B"], ["A", "B"])
    with pytest.raises(ValueError, match="unmeasured"):
        validate_measured_coordinates(matrix, presence, ["B", "A"], ["A", "B"])
