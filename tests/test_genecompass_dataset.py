import json

import numpy as np
import pytest

from dinogenept.cell.genecompass_data import GeneCompassDataset
from dinogenept.cell.sampling import CropConfig
from dinogenept.provenance import digest_file


def corpus(root):
    vocab = root / "vocabulary.json"
    vocab.write_text(json.dumps(["ENSG1", "ENSG2"]))
    ids = np.zeros((2, 2048), dtype=np.int32)
    values = np.zeros((2, 2048), dtype=np.float32)
    ids[:, :2] = [2, 1]
    values[:, :2] = [0.8, 0.3]
    shard = root / "shard.npz"
    np.savez(shard, gene_ids=ids, expression=values)
    manifest = root / "manifest.json"
    record = dict(schema="dinogenept.genecompass.continuous.v1",
                  protocol="genecompass_all_cells_one_epoch_no_validation",
                  expression="genecompass_published_continuous", downstream_overlap="unknown",
                  validation_cells=0, training_cells=2, data_id="fixture",
                  vocabulary=dict(path=vocab.name, genes=2, sha256=digest_file(vocab)),
                  shards=[dict(path=shard.name, cells=2, sha256=digest_file(shard))])
    manifest.write_text(json.dumps(record))
    return manifest, record


def test_continuous_loader_identity_and_scale(tmp_path):
    path, _ = corpus(tmp_path)
    data = GeneCompassDataset(path, CropConfig(global_scale=(1, 1)))
    assert data.verify() == digest_file(path)
    assert len(data) == 2
    assert data[0]["cell_id"] != data[1]["cell_id"]
    np.testing.assert_array_equal(data[0]["views"][0]["gene_ids"], [1, 2])
    np.testing.assert_array_equal(data[0]["views"][0]["expression"], np.array([0.3, 0.8], dtype=np.float32))
    assert not data.__getstate__()["cache"]
    with pytest.raises(IndexError):
        data[2]


def test_reject_changed_shard(tmp_path):
    path, _ = corpus(tmp_path)
    (tmp_path / "shard.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        GeneCompassDataset(path, CropConfig()).verify()


@pytest.mark.parametrize("field,value", [("downstream_overlap", "passed"), ("validation_cells", 1),
                                          ("expression", "raw_counts")])
def test_reject_misleading_protocol(tmp_path, field, value):
    path, record = corpus(tmp_path)
    record[field] = value
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="protocol"):
        GeneCompassDataset(path, CropConfig())


def test_reject_artifact_escape(tmp_path):
    path, record = corpus(tmp_path)
    record["vocabulary"]["path"] = "../escape.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="escapes"):
        GeneCompassDataset(path, CropConfig()).verify()
