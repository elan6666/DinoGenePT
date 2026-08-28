import json
import zipfile

from genept_seed.gradpert_union import DATASETS
from genept_seed.knowledge_corpus import audit_knowledge_corpora, build_knowledge_corpus


def _sources(tmp_path):
    uniprot = tmp_path / "uniprot.tsv"
    uniprot.write_text(
        "Entry\tGene Names (primary)\tProtein names\tSubcellular location [CC]\tInterPro\n"
        "P1\tA\tProtein A\tNucleus\tIPR1; IPR2;\n"
    )
    interpro = tmp_path / "interpro.tsv"
    interpro.write_text("ENTRY_AC\tENTRY_TYPE\tENTRY_NAME\nIPR1\tDomain\tKinase\n")
    reactome = tmp_path / "reactome.tsv"
    reactome.write_text("P1\tR-HSA-1\thttps://x\tSignal pathway\tTAS\tHomo sapiens\n")
    signor = tmp_path / "signor.tsv"
    signor.write_text(
        "ENTITYA\tTYPEA\tENTITYB\tTYPEB\tEFFECT\tMECHANISM\tTAX_ID\tDIRECT\n"
        "A\tprotein\tB\tprotein\tup-regulates\tphosphorylation\t9606\tt\n"
    )
    hpa = tmp_path / "hpa.zip"
    with zipfile.ZipFile(hpa, "w") as archive:
        archive.writestr("proteinatlas.tsv", "Gene\tRNA tissue specificity\nA\tTissue enriched\n")
    return uniprot, interpro, reactome, signor, hpa


def test_sparse_progressive_sections_preserve_complete_base(tmp_path):
    base, genes = tmp_path / "base.json", tmp_path / "genes.txt"
    base.write_text(json.dumps({"A": "base A", "B": "base B"}))
    genes.write_text("A\nB\n")
    uniprot, interpro, reactome, signor, hpa = _sources(tmp_path)
    receipts = []
    outputs = []
    for profile in ("protein", "protein-pathway", "protein-pathway-hpa"):
        output = tmp_path / f"{profile}.json"
        receipts.append(
            build_knowledge_corpus(
                base_path=base,
                genes_path=genes,
                uniprot_path=uniprot,
                interpro_path=interpro,
                reactome_path=reactome,
                signor_path=signor,
                hpa_path=hpa,
                profile=profile,
                output_path=output,
                manifest_path=tmp_path / f"{profile}.manifest.json",
            )
        )
        outputs.append(json.loads(output.read_text()))
    assert all(set(output) == {"A", "B"} for output in outputs)
    assert outputs[0]["B"] == "base B"
    assert "InterPro entries" in outputs[0]["A"]
    assert "Reactome pathways" not in outputs[0]["A"]
    assert "Reactome pathways" in outputs[1]["A"]
    assert "SIGNOR direct causal relations" in outputs[1]["B"]
    assert "Human Protein Atlas" in outputs[2]["A"]
    assert all(receipt["all_genes_preserved"] for receipt in receipts)


def test_audit_knowledge_corpora_proves_append_only_graph_and_target_coverage(tmp_path):
    genes = tmp_path / "genes.txt"
    genes.write_text("A\nB\n")
    paths = []
    for index, payload in enumerate(
        (
            {"A": "a", "B": "b"},
            {"A": "a protein", "B": "b"},
            {"A": "a protein pathway", "B": "b"},
            {"A": "a protein pathway hpa", "B": "b"},
        )
    ):
        path = tmp_path / f"corpus-{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(path)
    gradpert = tmp_path / "gradpert"
    for dataset, protocol in DATASETS:
        root = gradpert / dataset / protocol
        (root / "canonical").mkdir(parents=True)
        (root / "manifests").mkdir(parents=True)
        (root / "canonical" / "graph_gene_ids.txt").write_text("A\nB\n")
        (root / "manifests" / "split.json").write_text(
            json.dumps(
                {
                    "control_condition_id": "ctrl",
                    "train_conditions": ["A+ctrl"],
                    "val_conditions": [],
                    "test_conditions": [],
                }
            )
        )
    receipt = audit_knowledge_corpora(
        genes_path=genes,
        gradpert_root=gradpert,
        base_path=paths[0],
        protein_path=paths[1],
        pathway_path=paths[2],
        hpa_path=paths[3],
        output_path=tmp_path / "audit.json",
    )
    assert receipt["all_graph_axes_covered"] is True
    assert receipt["all_perturbation_targets_covered"] is True
    assert receipt["transitions"]["seed_go_to_seed_go_protein"] == {
        "changed_genes": 1,
        "unchanged_genes": 1,
        "all_previous_text_is_prefix": True,
    }
