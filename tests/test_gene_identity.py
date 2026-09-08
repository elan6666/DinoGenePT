import csv
import json

import numpy as np
import pytest
from test_knowledge_bank import make_artifacts

from dinogenept.cli import main
from dinogenept.datasets.identity_knowledge import IdentityKnowledgeBank
from dinogenept.datasets.knowledge import KnowledgeBank
from dinogenept.gene_identity import GeneIdentityIndex, build_identity_report, read_identity_json
from dinogenept.provenance import digest_file
from dinogenept.vectors import load_npz


@pytest.fixture
def snapshot(tmp_path):
    path = tmp_path / "hgnc.tsv"
    fields = ["hgnc_id", "symbol", "status", "prev_symbol", "alias_symbol", "ensembl_gene_id"]
    with path.open("w") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(fields)
        writer.writerows([
            ["HGNC:1", "A", "Approved", "OLD_A", "SHARED|B", "ENSG00000000001"],
            ["HGNC:2", "B", "Approved", "OLD_B", "SHARED", "ENSG00000000002"],
            ["HGNC:3", "C", "Approved", "", "", "ENSG00000000003"],
            ["HGNC:4", "DEAD", "Entry Withdrawn", "", "", "ENSG00000000004"],
        ])
    return path


def index(snapshot):
    return GeneIdentityIndex(snapshot, expected_sha256=digest_file(snapshot))


@pytest.mark.parametrize("label,id_,status,hgnc", [
    ("A", None, "resolved", "HGNC:1"),
    ("old_a", None, "resolved", "HGNC:1"),
    ("B", None, "resolved", "HGNC:2"),  # Approved name wins over alias.
    ("SHARED", None, "ambiguous", None),
    ("SHARED", "ENSG00000000002.7", "resolved", "HGNC:2"),
    ("A", "ENSG00000000002", "conflict", None),
    ("A", "ENSG00000009999", "unverified_ensembl_id", None),
    ("A", "ENSMUSG00000000001", "invalid_ensembl_id", None),
    ("ENSG00000000001.2", None, "resolved", "HGNC:1"),
    ("ENSG00000000001", "ENSG00000000002", "conflict", None),
    ("unknown", "ENSG00000000003", "resolved", "HGNC:3"),
    ("DEAD", None, "unresolved", None),
    ("unknown", None, "unresolved", None),
])
def test_conservative_resolution(snapshot, label, id_, status, hgnc):
    row = index(snapshot).resolve(label, id_)
    assert row["status"] == status
    assert row["hgnc_id"] == hgnc
    assert row["original_label"] == label


def test_axis_preserves_order_and_does_not_merge_duplicate_identities(snapshot):
    result = index(snapshot).axis(["B", "OLD_A", "A", "B"])
    assert [r["original_label"] for r in result["rows"]] == ["B", "OLD_A", "A", "B"]
    assert result["duplicate_identities"] == {"HGNC:2": [0, 3], "HGNC:1": [1, 2]}
    assert result["duplicate_labels"] == {"B": [0, 3]}
    assert not result["safe_unique_axis"]
    with pytest.raises(ValueError, match="checksum"):
        GeneIdentityIndex(snapshot, expected_sha256="bad")


def test_sources_share_identity_but_preserve_text_conflicts_and_missing(snapshot):
    result = build_identity_report(index(snapshot), ["OLD_A", "B", "unknown"], {
        "TextBase": {"A": "unchanged", "OLD_A": "unchanged", "B": "base B"},
        "GO": {"A": "GO A"},
        "Protein": {"A": "v1", "OLD_A": "v2"},
    })
    joins = result["sources"]
    assert joins["TextBase"]["joins"][0]["source_key"] == "A"
    assert joins["TextBase"]["joins"][0]["equivalent_keys"] == ["A", "OLD_A"]
    assert joins["GO"]["joins"][1]["status"] == "no_resolved_source_record"
    assert joins["GO"]["joins"][2]["status"] == "identity_unresolved"
    assert joins["Protein"]["joins"][0]["status"] == "source_conflict"


def test_source_unknown_name_is_not_claimed_absent_annotation(snapshot):
    result = build_identity_report(index(snapshot), ["A"], {"GO": {"SHARED": "term"}})
    assert result["sources"]["GO"]["unresolved_source_keys"][0]["status"] == "ambiguous"
    assert result["sources"]["GO"]["joins"][0]["status"] == "source_identity_ambiguous"


def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"A":"first", "A":"second"}')
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        read_identity_json(path)


def test_adapter_rejects_conflicting_text_variants(snapshot, tmp_path):
    sources = make_artifacts(tmp_path / "vectors", genes=("A", "OLD_A"))
    with pytest.raises(ValueError, match="Conflicting source texts"):
        IdentityKnowledgeBank(hgnc=snapshot, hgnc_sha256=digest_file(snapshot),
                              axis_records=["A"], sources=sources, model="fixture-model", width=6)


def test_native_adapter_reuses_verified_vectors_in_axis_order(snapshot, tmp_path):
    sources = make_artifacts(tmp_path / "vectors")
    # Give the two rows distinct values, retaining a consistent artifact receipt.
    from pathlib import Path

    source = sources["TextBase"]
    p = Path(source["vectors"]["path"])
    original = load_npz(p)
    original.vectors[1] *= 2
    np.savez_compressed(p, genes=original.genes, vectors=original.vectors, model=np.asarray(original.model))
    source["vectors"]["sha256"] = digest_file(p)
    manifest_path = Path(source["embedding_manifest"]["path"])
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"] = digest_file(p)
    manifest_path.write_text(json.dumps(manifest))
    source["embedding_manifest"]["sha256"] = digest_file(manifest_path)
    bank = IdentityKnowledgeBank(hgnc=snapshot, hgnc_sha256=digest_file(snapshot),
                                 axis_records=["OLD_B", "OLD_A"], sources=sources, model="fixture-model", width=6)
    assert bank.genes == ("OLD_B", "OLD_A")
    np.testing.assert_array_equal(bank.for_axis()[1], original.vectors[[1, 0]])
    positions, vectors = bank.for_axis("GO")
    np.testing.assert_array_equal(positions, [1])
    assert vectors.shape == (1, 6)
    assert bank.for_targets(["OLD_B", "OLD_A"])["GO"] is None
    assert bank.for_axis("HPA")[1].shape == (0, 6)
    assert bank.for_targets(["OLD_A"])["GO"].shape == (1, 6)
    with pytest.raises(ValueError, match="duplicate identities"):
        IdentityKnowledgeBank(hgnc=snapshot, hgnc_sha256=digest_file(snapshot),
                              axis_records=["A", "OLD_A"], sources=sources, model="fixture-model", width=6)
    with pytest.raises(ValueError, match="Required TextBase missing"):
        IdentityKnowledgeBank(hgnc=snapshot, hgnc_sha256=digest_file(snapshot),
                              axis_records=["C"], sources=sources, model="fixture-model", width=6)
    with pytest.raises(ValueError, match="Embedding provenance"):
        IdentityKnowledgeBank(hgnc=snapshot, hgnc_sha256=digest_file(snapshot),
                              axis_records=["A"], sources=sources, model="wrong", width=6)


def test_cli_creates_fresh_audit_and_never_rewrites_inputs(snapshot, tmp_path, capsys):
    axis, sources, output = [tmp_path / name for name in ["axis.json", "sources.json", "report.json"]]
    axis.write_text(json.dumps(["OLD_A", "SHARED"]))
    sources.write_text("{}")
    before = digest_file(axis)
    args = ["data", "audit-gene-identities", "--hgnc", str(snapshot), "--hgnc-sha256", digest_file(snapshot),
            "--axis", str(axis), "--sources", str(sources), "--output", str(output)]
    assert main(args) == 0  # Successful AUDIT, not a claim that all genes resolved.
    assert json.loads(capsys.readouterr().out)["counts"] == {"resolved": 1, "ambiguous": 1}
    assert digest_file(axis) == before
    with pytest.raises(FileExistsError):
        main(args)


def test_existing_loader_opt_in_validates_frozen_axis_order(snapshot, tmp_path):
    sources = make_artifacts(tmp_path / "vectors")
    path = tmp_path / "axis.json"
    path.write_text(json.dumps(["OLD_B", "OLD_A"]))
    config = {"hgnc": str(snapshot), "hgnc_sha256": digest_file(snapshot),
              "axis": {"path": str(path), "sha256": digest_file(path)}}
    bank = KnowledgeBank(sources, ["OLD_B", "OLD_A"], model="fixture-model", width=6, identity=config)
    assert bank.for_targets(["OLD_A"])["GO"].shape == (1, 6)
    assert bank.for_targets(["OLD_B"])["GO"] is None
    with pytest.raises(ValueError, match="axis/order"):
        KnowledgeBank(sources, ["OLD_A", "OLD_B"], model="fixture-model", width=6, identity=config)
    path.write_text(json.dumps(["A", "B"]))
    with pytest.raises(ValueError, match="checksum"):
        KnowledgeBank(sources, ["OLD_B", "OLD_A"], model="fixture-model", width=6, identity=config)
