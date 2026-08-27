import numpy as np
import pytest

from genept_seed.vectors import coverage, l2_normalize, load_npz, load_official_pickle, save_npz


def test_npz_roundtrip_and_normalization(tmp_path):
    path = tmp_path / "vectors.npz"
    save_npz(path, {"b": np.array([0, 2]), "a": np.array([3, 4])}, "mock")
    loaded = load_npz(path)
    assert loaded.genes.tolist() == ["A", "B"]
    normalized = l2_normalize(loaded)
    assert np.allclose(np.linalg.norm(normalized.vectors, axis=1), 1)
    assert coverage(loaded, {"A", "C"})["coverage"] == 0.5


def test_pickle_requires_explicit_trust(tmp_path):
    with pytest.raises(ValueError, match="trusted=True"):
        load_official_pickle(tmp_path / "official.pickle")
