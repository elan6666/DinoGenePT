from genept_seed.property_data import _clean_binary, _dosage, _hgnc_ensembl_map, _long_short


def test_property_source_parsers_and_hgnc_mapping(tmp_path):
    hgnc = tmp_path / "hgnc.tsv"
    hgnc.write_text("symbol\tensembl_gene_id\nTFAP2D\tENSG00000008197\n", encoding="utf-8")
    assert _hgnc_ensembl_map(hgnc)["ENSG00000008197"] == "TFAP2D"

    dosage = tmp_path / "dosage.csv"
    dosage.write_text(
        "dosage_sensitive,dosage_insensitive\nENSG00000008197,ENSG00000010539\n",
        encoding="utf-8",
    )
    assert _dosage(dosage) == (["ENSG00000008197"], ["ENSG00000010539"])

    long_short = tmp_path / "long-short.csv"
    long_short.write_text(",assignment\nA,long-range TF\nB,short-range TF\n", encoding="utf-8")
    assert _long_short(long_short) == (["A"], ["B"])


def test_property_binary_overlap_is_removed_from_both_classes():
    positive, negative, overlap = _clean_binary(["A", "B"], ["B", "C"])
    assert positive == ["A"]
    assert negative == ["C"]
    assert overlap == ["B"]
