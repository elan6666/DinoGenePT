import pickle

import numpy as np
import pytest

from dinogenept.axis_vectors import align_npz_to_axis, materialize_axis_vectors


def test_materialize_axis_vectors_uses_alias_and_zero_fallback(tmp_path):
    source = tmp_path / "source.pickle"
    genes = tmp_path / "genes.txt"
    hgnc = tmp_path / "hgnc.tsv"
    output = tmp_path / "vectors.npz"
    with source.open("wb") as handle:
        pickle.dump({"TP53": [1.0, 2.0], "NEW1": [3.0, 4.0]}, handle)
    genes.write_text("TP53\nOld1\nMISS\n")
    hgnc.write_text(
        "hgnc_id\tsymbol\tname\tlocus_group\tlocus_type\tstatus\tlocation\talias_symbol\t"
        "prev_symbol\tentrez_id\tensembl_gene_id\tuniprot_ids\n"
        "HGNC:1\tNEW1\tnew gene\tprotein-coding gene\tgene with protein product\tApproved\t1p1\t\t"
        "Old1\t1\tENSG00000100001\tP1\n"
    )
    receipt = materialize_axis_vectors(
        source_path=source,
        genes_path=genes,
        hgnc_path=hgnc,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        model="official-test",
        trusted_pickle=True,
    )
    with np.load(output, allow_pickle=False) as data:
        assert data["genes"].tolist() == ["TP53", "Old1", "MISS"]
        assert data["vectors"].tolist() == [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]
    assert receipt["source_counts"] == {
        "exact_official": 1,
        "hgnc_alias_official": 1,
        "learned_id_fallback": 1,
    }


def test_align_npz_to_axis_preserves_exact_requested_order(tmp_path):
    source = tmp_path / "source.npz"
    genes = tmp_path / "genes.txt"
    output = tmp_path / "aligned.npz"
    np.savez_compressed(
        source,
        genes=np.asarray(["A", "B"]),
        vectors=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        model=np.asarray("test-model"),
    )
    genes.write_text("B\nA\n", encoding="utf-8")
    receipt = align_npz_to_axis(
        source_path=source,
        genes_path=genes,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
    )
    with np.load(output, allow_pickle=False) as data:
        assert data["genes"].tolist() == ["B", "A"]
        assert data["vectors"].tolist() == [[3.0, 4.0], [1.0, 2.0]]
    assert receipt["genes"] == 2
    assert receipt["dimension"] == 2


def test_align_npz_to_axis_rejects_incomplete_or_extra_axis(tmp_path):
    source = tmp_path / "source.npz"
    genes = tmp_path / "genes.txt"
    np.savez_compressed(
        source,
        genes=np.asarray(["A", "EXTRA"]),
        vectors=np.ones((2, 2), dtype=np.float32),
        model=np.asarray("test-model"),
    )
    genes.write_text("A\nMISSING\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing=.*MISSING.*extra=.*EXTRA"):
        align_npz_to_axis(
            source_path=source,
            genes_path=genes,
            output_path=tmp_path / "aligned.npz",
            manifest_path=tmp_path / "manifest.json",
        )
