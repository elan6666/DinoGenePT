import json

import pytest

from genept_seed.corpus import (
    NCBIGeneRecord,
    UniProtRecord,
    clean_uniprot_function,
    compose_genept_text,
    extend_genept_texts,
)


class FakeNCBI:
    def resolve(self, symbol, expected_gene_id=None):
        assert expected_gene_id in {None, "121053"}
        return NCBIGeneRecord(
            requested_symbol=symbol,
            current_symbol="NOPCHAP1",
            gene_id="121053",
            description="NOP protein chaperone 1",
            summary="Promotes box C/D snoRNP assembly.",
        )


class FakeUniProt:
    def find(self, symbol):
        assert symbol == "NOPCHAP1"
        return UniProtRecord(
            accession="Q8N5I9",
            reviewed=True,
            primary_symbol=symbol,
            protein_name="NOP protein chaperone 1",
            function="Selects NOP58 for snoRNP assembly.",
        )


def test_clean_uniprot_function_removes_labels_and_evidence():
    raw = "FUNCTION: First role. {ECO:0000269|PubMed:1}.; FUNCTION: Second role."
    assert clean_uniprot_function(raw) == "First role. Second role."


def test_compose_text_records_alias_and_sources():
    ncbi = FakeNCBI().resolve("C12ORF45")
    text = compose_genept_text(ncbi, FakeUniProt().find("NOPCHAP1"))
    assert text.startswith("Gene Symbol C12ORF45 Current NCBI symbol: NOPCHAP1.")
    assert "Protein summary: Selects NOP58" in text


def test_extend_genept_texts_merges_without_overwriting(tmp_path):
    base = tmp_path / "base.json"
    genes = tmp_path / "genes.txt"
    output = tmp_path / "extended.json"
    manifest = tmp_path / "manifest.json"
    base.write_text(json.dumps({"TP53": "Gene Symbol TP53 text"}))
    genes.write_text("C12orf45\t121053\n")
    receipt = extend_genept_texts(
        base_path=base,
        genes_path=genes,
        output_path=output,
        manifest_path=manifest,
        ncbi_client=FakeNCBI(),
        uniprot_client=FakeUniProt(),
    )
    merged = json.loads(output.read_text())
    assert set(merged) == {"TP53", "C12orf45"}
    assert receipt["added_entries"] == 1
    assert receipt["records"][0]["current_ncbi_symbol"] == "NOPCHAP1"
    assert json.loads(manifest.read_text())["output_sha256"] == receipt["output_sha256"]


def test_extend_refuses_existing_symbol(tmp_path):
    base = tmp_path / "base.json"
    genes = tmp_path / "genes.txt"
    base.write_text(json.dumps({"TP53": "text"}))
    genes.write_text("tp53\n")
    with pytest.raises(ValueError, match="overwrite"):
        extend_genept_texts(
            base_path=base,
            genes_path=genes,
            output_path=tmp_path / "out.json",
            manifest_path=tmp_path / "manifest.json",
            ncbi_client=FakeNCBI(),
            uniprot_client=FakeUniProt(),
        )
