import gzip
import json

from genept_seed.go_corpus import (
    SAFE_EXPERIMENTAL_EVIDENCE,
    build_go_exp_corpus,
    load_go_annotations,
    load_go_ontology,
)

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: parent process
namespace: biological_process
def: "Broad process." []

[Term]
id: GO:0000002
name: specific repair process
namespace: biological_process
def: "Repairs a specific lesion." []
is_a: GO:0000001 ! parent process

[Term]
id: GO:0000003
name: binding activity
namespace: molecular_function
def: "Binds a target molecule." []

[Term]
id: GO:0000004
name: nucleus
namespace: cellular_component
def: "A membrane-bounded organelle." []
"""


def _write_gaf(path):
    def row(symbol, qualifier, go_id, evidence, aspect, synonyms="", with_from=""):
        return [
            "UniProtKB",
            f"P-{symbol}",
            symbol,
            qualifier,
            go_id,
            "PMID:1",
            evidence,
            with_from,
            aspect,
            "",
            synonyms,
            "protein",
            "taxon:9606",
            "20260101",
            "UniProt",
        ]

    rows = [
        row("TP53", "", "GO:0000001", "EXP", "P"),
        row("TP53", "", "GO:0000002", "IDA", "P"),
        row("TP53", "", "GO:0000003", "IPI", "F", with_from="P2"),
        row("OTHER", "", "GO:0000004", "HDA", "C", synonyms="BRCA1"),
        row("OTHER", "NOT", "GO:0000003", "IDA", "F", synonyms="BRCA1"),
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("!gaf-version: 2.2\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def test_go_parsing_filters_interaction_not_and_prunes_ancestors(tmp_path):
    obo = tmp_path / "go.obo"
    gaf = tmp_path / "human.gaf.gz"
    obo.write_text(OBO)
    _write_gaf(gaf)
    ontology = load_go_ontology(obo)
    annotations, stats = load_go_annotations(
        gaf,
        target_genes={"TP53", "BRCA1"},
        evidence_codes=SAFE_EXPERIMENTAL_EVIDENCE,
    )
    assert len(ontology) == 4
    assert {row.go_id for row in annotations["TP53"]} == {"GO:0000001", "GO:0000002"}
    assert annotations["BRCA1"][0].go_id == "GO:0000004"
    assert stats["excluded_IPI"] == 1
    assert stats["excluded_not"] == 1


def test_build_go_exp_corpus_keeps_universe_and_adds_bounded_text(tmp_path):
    base = tmp_path / "base.json"
    genes = tmp_path / "genes.txt"
    obo = tmp_path / "go.obo"
    gaf = tmp_path / "human.gaf.gz"
    output = tmp_path / "output.json"
    manifest = tmp_path / "manifest.json"
    base.write_text(json.dumps({"TP53": "tp53 base", "BRCA1": "brca1 base"}))
    genes.write_text("TP53\nBRCA1\n")
    obo.write_text(OBO)
    _write_gaf(gaf)
    receipt = build_go_exp_corpus(
        base_path=base,
        genes_path=genes,
        gaf_path=gaf,
        obo_path=obo,
        output_path=output,
        manifest_path=manifest,
        max_terms_per_aspect=1,
    )
    enriched = json.loads(output.read_text())
    assert set(enriched) == {"TP53", "BRCA1"}
    assert "specific repair process" in enriched["TP53"]
    assert "parent process" not in enriched["TP53"]
    assert "binding activity" not in enriched["TP53"]
    assert "nucleus" in enriched["BRCA1"]
    assert receipt["genes"] == 2
    assert receipt["selection_stats"]["genes_with_go"] == 2
    assert json.loads(manifest.read_text())["interaction_evidence_included"] is False


def test_build_go_exp_corpus_preserves_axis_case_and_order(tmp_path):
    base = tmp_path / "base.json"
    genes = tmp_path / "genes.txt"
    obo = tmp_path / "go.obo"
    gaf = tmp_path / "human.gaf.gz"
    output = tmp_path / "output.json"
    base.write_text(json.dumps({"C12ORF45": "mixed-case base", "TP53": "tp53 base"}))
    genes.write_text("C12orf45\nTP53\n")
    obo.write_text(OBO)
    _write_gaf(gaf)
    build_go_exp_corpus(
        base_path=base,
        genes_path=genes,
        gaf_path=gaf,
        obo_path=obo,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
    )
    enriched = json.loads(output.read_text())
    assert list(enriched) == ["C12orf45", "TP53"]
    assert enriched["C12orf45"] == "mixed-case base"
