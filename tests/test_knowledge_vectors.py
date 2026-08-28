import json

import numpy as np

from genept_seed.gradpert_union import DATASETS
from genept_seed.knowledge_vectors import audit_knowledge_vectors
from genept_seed.vectors import save_npz


def test_audit_knowledge_vectors_proves_exact_coverage(tmp_path):
    genes = tmp_path / "genes.txt"
    ggi = tmp_path / "ggi.txt"
    genes.write_text("A\nB\nC12ORF57\nC12orf57\n")
    ggi.write_text("A\nC12ORF57\n")
    vectors = {
        "A": np.asarray([1, 0], dtype=np.float32),
        "B": np.asarray([0, 1], dtype=np.float32),
        "C12ORF57": np.asarray([1, 1], dtype=np.float32),
        "C12orf57": np.asarray([1, 1], dtype=np.float32),
    }
    paths = []
    for index in range(3):
        path = tmp_path / f"vectors-{index}.npz"
        save_npz(path, vectors, "mock", uppercase_genes=False)
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
    receipt = audit_knowledge_vectors(
        genes_path=genes,
        ggi_genes_path=ggi,
        gradpert_root=gradpert,
        protein_path=paths[0],
        pathway_path=paths[1],
        hpa_path=paths[2],
        output_path=tmp_path / "audit.json",
        expected_dimension=2,
    )
    assert receipt["all_graph_axes_exact_coverage"] is True
    assert receipt["all_targets_exact_coverage"] is True
    assert receipt["conditions"]["seed_go_protein"]["ggi_exact_labels"] == 2
    assert receipt["casefold_collision_groups"] == {
        "C12ORF57": ["C12ORF57", "C12orf57"]
    }
