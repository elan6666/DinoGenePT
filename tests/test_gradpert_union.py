import json

from genept_seed.gradpert_union import DATASETS, build_gradpert_union


def test_build_gradpert_union_includes_graph_targets_and_extras(tmp_path):
    root = tmp_path / "gradpert"
    for index, (dataset, protocol) in enumerate(DATASETS):
        target = f"T{index}"
        folder = root / dataset / protocol
        (folder / "canonical").mkdir(parents=True)
        (folder / "manifests").mkdir(parents=True)
        (folder / "canonical" / "graph_gene_ids.txt").write_text(f"COMMON\n{target}\n")
        (folder / "manifests" / "split.json").write_text(
            json.dumps(
                {
                    "control_condition_id": "ctrl",
                    "train_conditions": [f"{target}+ctrl"],
                    "val_conditions": [],
                    "test_conditions": [],
                }
            )
        )
    extra = tmp_path / "extra.txt"
    extra.write_text("GGI1\nCOMMON\n")
    output = tmp_path / "master.txt"
    receipt = build_gradpert_union(
        gradpert_root=root,
        extra_genes_path=extra,
        output_path=output,
        manifest_path=tmp_path / "manifest.json",
    )
    assert receipt["graph_union_genes"] == 6
    assert receipt["perturbation_target_union_genes"] == 5
    assert receipt["master_genes"] == 7
    assert receipt["all_targets_in_graph_union"] is True
    assert output.read_text().splitlines() == sorted(output.read_text().splitlines())
