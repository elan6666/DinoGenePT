import numpy as np

from genept_seed.benchmarks import evaluate_ggi, evaluate_property_task
from genept_seed.tasks import GGIDataset, ggi_genes, load_property_tasks


def test_property_parser_and_benchmark(tmp_path):
    path = tmp_path / "tasks.csv"
    rows = ["task,gene,label"]
    vectors = {}
    genes = []
    labels = []
    for index in range(20):
        gene = f"G{index}"
        label = index % 2
        rows.append(f"demo,{gene},{label}")
        genes.append(gene)
        labels.append(label)
        vectors[gene] = np.asarray([label, index / 20], dtype=np.float32)
    path.write_text("\n".join(rows))
    task = load_property_tasks(path)["demo"]
    results = evaluate_property_task(task, vectors, folds=5)
    assert len(results) == 10
    assert {row["model"] for row in results} == {"logistic_regression", "random_forest"}


def test_fixed_split_ggi_benchmark():
    vectors = {
        "A": np.asarray([0.0, 0.0]),
        "B": np.asarray([0.0, 1.0]),
        "C": np.asarray([1.0, 0.0]),
        "D": np.asarray([1.0, 1.0]),
    }
    dataset = GGIDataset(
        train_pairs=[("A", "B"), ("C", "D"), ("A", "A"), ("D", "D")],
        train_labels=[0, 1, 0, 1],
        test_pairs=[("A", "B"), ("C", "D")],
        test_labels=[0, 1],
    )
    result = evaluate_ggi(dataset, vectors)
    assert result["pair_operator"] == "sum"
    assert result["test_coverage"] == 1.0
    assert ggi_genes(dataset) == {"A", "B", "C", "D"}
