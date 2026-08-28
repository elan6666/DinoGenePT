import pickle

import numpy as np

from genept_seed.axis_vectors import materialize_axis_vectors


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
