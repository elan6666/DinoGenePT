import numpy as np
import pytest

from genept_seed.vectors import (
    EmbeddingSet,
    coverage,
    l2_normalize,
    load_npz,
    load_official_pickle,
    save_npz,
    select_universe_vectors,
)


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


def test_select_universe_vectors_prefers_exact_case_over_casefold_collision():
    loaded = EmbeddingSet(
        genes=np.asarray(["C12ORF57", "C12orf57", "TP53"]),
        vectors=np.asarray([[1, 0], [0, 1], [2, 2]], dtype=np.float32),
        model="mock",
    )
    selected = select_universe_vectors(loaded, {"C12ORF57", "TP53"})
    assert np.array_equal(selected["C12ORF57"], np.asarray([1, 0], dtype=np.float32))
    assert np.array_equal(selected["TP53"], np.asarray([2, 2], dtype=np.float32))
