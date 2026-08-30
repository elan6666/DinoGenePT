import json

from genept_seed.experiment_comparison import (
    summarize_gene_disjoint_results,
    summarize_property_results,
)


def test_property_summary_enforces_shared_contract(tmp_path):
    paths = []
    for embedding in ("a", "b"):
        rows = []
        for seed in (1, 2):
            for fold in (1, 2):
                rows.append(
                    {
                        "embedding": embedding,
                        "task": "t",
                        "model": "logistic_regression",
                        "fold": fold,
                        "random_state": seed,
                        "n": 20,
                        "normalization": "l2",
                        "gene_universe_sha256": "u",
                        "data_receipt_sha256": "d",
                        "genept_seed_version": "v",
                        "numpy_version": "n",
                        "scikit_learn_version": "s",
                        "accuracy": 0.5,
                        "auroc": 0.6,
                        "average_precision": 0.7,
                    }
                )
        path = tmp_path / f"{embedding}.json"
        path.write_text(json.dumps(rows))
        paths.append(path)
    receipt = summarize_property_results(result_paths=paths, output_path=tmp_path / "out.json")
    assert receipt["fairness_fields_identical"] is True
    assert len(receipt["conditions"]) == 2


def test_gene_disjoint_summary_enforces_identical_splits(tmp_path):
    paths = []
    for embedding in ("a", "b"):
        rows = []
        for seed in (1, 2):
            rows.append(
                {
                    "embedding": embedding,
                    "random_state": seed,
                    "split_receipt": {"seed": seed, "test_gene_sha256": str(seed)},
                    "universe_filter": {
                        "requested_genes": 20,
                        "all_pairs_within_universe": True,
                    },
                    "normalization": "l2",
                    "gene_universe_sha256": "u",
                    "data_receipt_sha256": "d",
                    "model": "logistic_regression",
                    "pair_operator": "sum",
                    "genept_seed_version": "v",
                    "numpy_version": "n",
                    "scikit_learn_version": "s",
                    "vectors_sha256": embedding,
                    "accuracy": 0.5,
                    "auroc": 0.6,
                    "average_precision": 0.7,
                }
            )
        path = tmp_path / f"{embedding}.json"
        path.write_text(json.dumps(rows))
        paths.append(path)
    receipt = summarize_gene_disjoint_results(result_paths=paths, output_path=tmp_path / "out.json")
    assert receipt["split_receipts_identical"] is True
    assert receipt["universe_filters_identical"] is True
    assert len(receipt["conditions"]) == 2
