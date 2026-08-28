import json

from genept_seed.axis_corpus import build_axis_corpus


def test_build_axis_corpus_preserves_order_and_uses_hgnc_fallback(tmp_path):
    source = tmp_path / "source.json"
    genes = tmp_path / "genes.txt"
    hgnc = tmp_path / "hgnc.tsv"
    mapping = tmp_path / "mapping.tsv"
    output = tmp_path / "output.json"
    source.write_text(json.dumps({"TP53": "Gene Symbol TP53 base", "NEW1": "Gene Symbol NEW1 base"}))
    genes.write_text("TP53\nOld1\nLNC1\n")
    mapping.write_text("LNC1\t-\tENSG00000200001\n")
    hgnc.write_text(
        "hgnc_id\tsymbol\tname\tlocus_group\tlocus_type\tstatus\tlocation\talias_symbol\t"
        "prev_symbol\tentrez_id\tensembl_gene_id\tuniprot_ids\n"
        "HGNC:1\tNEW1\tnew gene\tprotein-coding gene\tgene with protein product\tApproved\t1p1\t\t"
        "Old1\t1\tENSG00000100001\tP1\n"
        "HGNC:2\tLNCX\tlong RNA\tnon-coding RNA\tRNA, long non-coding\tApproved\t2q2\t\t\t\t"
        "ENSG00000200001\t\n"
    )
    receipt = build_axis_corpus(
        source_path=source,
        genes_path=genes,
        hgnc_path=hgnc,
        axis_mapping_path=mapping,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
    )
    corpus = json.loads(output.read_text())
    assert set(corpus) == {"TP53", "Old1", "LNC1"}
    assert [record["gene"] for record in receipt["records"]] == ["TP53", "Old1", "LNC1"]
    assert "Current HGNC symbol: NEW1" in corpus["Old1"]
    assert "HGNC locus type: RNA, long non-coding" in corpus["LNC1"]
    assert receipt["source_counts"] == {
        "exact_genept": 1,
        "hgnc_alias_to_genept": 1,
        "hgnc_identity": 1,
    }


def test_build_axis_corpus_can_preserve_unknown_axis_identity_without_invented_function(tmp_path):
    source = tmp_path / "source.json"
    genes = tmp_path / "genes.txt"
    hgnc = tmp_path / "hgnc.tsv"
    source.write_text(json.dumps({"A": "base A"}))
    genes.write_text("A\nRP11-UNKNOWN.1\n")
    hgnc.write_text(
        "hgnc_id\tsymbol\tname\tlocus_group\tlocus_type\tstatus\tlocation\talias_symbol\t"
        "prev_symbol\tentrez_id\tensembl_gene_id\tuniprot_ids\n"
    )
    output = tmp_path / "output.json"
    receipt = build_axis_corpus(
        source_path=source,
        genes_path=genes,
        hgnc_path=hgnc,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
        allow_identity_only=True,
    )
    assert json.loads(output.read_text())["RP11-UNKNOWN.1"] == "Gene Symbol RP11-UNKNOWN.1"
    assert receipt["source_counts"]["axis_identity_only"] == 1
