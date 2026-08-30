import numpy as np

from genept_seed.benchmarks import evaluate_ggi, evaluate_property_task, evaluate_property_task_repeated
from genept_seed.ggi_controls import (
    audit_signor_ggi_leakage,
    filter_ggi_universe,
    gene_disjoint_split,
)
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
    repeated = evaluate_property_task_repeated(task, vectors, folds=5, seeds=(1, 2))
    assert len(repeated) == 20
    assert {row["random_state"] for row in repeated} == {1, 2}


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


def test_gene_disjoint_split_has_no_gene_overlap():
    genes = [f"G{i}" for i in range(20)]
    pairs, labels = [], []
    for i, left in enumerate(genes):
        for j, right in enumerate(genes[i:], start=i):
            pairs.append((left, right))
            labels.append((i + j) % 2)
    dataset = GGIDataset(pairs[:100], labels[:100], pairs[100:], labels[100:])
    split, receipt = gene_disjoint_split(dataset, seed=42, test_fraction=0.3)
    train_genes = {gene for pair in split.train_pairs for gene in pair}
    test_genes = {gene for pair in split.test_pairs for gene in pair}
    assert train_genes.isdisjoint(test_genes)
    assert receipt["gene_overlap"] == 0


def test_ggi_universe_is_filtered_before_gene_partition():
    dataset = GGIDataset(
        train_pairs=[("A", "B"), ("A", "X")],
        train_labels=[1, 0],
        test_pairs=[("A", "B"), ("B", "X")],
        test_labels=[0, 1],
    )
    filtered, receipt = filter_ggi_universe(dataset, {"A", "B"})
    assert filtered.train_pairs == [("A", "B")]
    assert filtered.test_pairs == [("A", "B")]
    assert receipt["requested_genes"] == 2
    assert receipt["all_pairs_within_universe"] is True
    assert receipt["isolated_genes"] == 0


def test_gene_disjoint_partition_keeps_isolated_allowlist_genes():
    connected = [f"G{i}" for i in range(10)]
    pairs = [
        (left, right)
        for index, left in enumerate(connected)
        for right in connected[index:]
    ]
    labels = [
        (int(left[1:]) + int(right[1:])) % 2
        for left, right in pairs
    ]
    dataset = GGIDataset(
        train_pairs=pairs[:30],
        train_labels=labels[:30],
        test_pairs=pairs[30:],
        test_labels=labels[30:],
    )
    for seed in range(100):
        try:
            _, receipt = gene_disjoint_split(
                dataset,
                seed=seed,
                test_fraction=0.3,
                universe={*connected, "ISOLATED"},
            )
        except ValueError:
            continue
        assert receipt["genes"] == 11
        assert receipt["train_genes"] + receipt["test_genes"] == 11
        break
    else:
        raise AssertionError("no test seed retained both labels")


def test_signor_leakage_distinguishes_source_from_selected_text(tmp_path):
    signor = tmp_path / "signor.tsv"
    signor.write_text(
        "ENTITYA\tTYPEA\tENTITYB\tTYPEB\tTAX_ID\tDIRECT\n"
        "A\tprotein\tB\tprotein\t9606\tYES\n"
        "C\tprotein\tD\tprotein\t9606\tYES\n",
        encoding="utf-8",
    )
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '{"A":"base\\nSIGNOR direct causal relations: outgoing: this protein up-regulates B",'
        '"B":"base","C":"base","D":"base"}',
        encoding="utf-8",
    )
    dataset = GGIDataset(
        train_pairs=[("A", "B"), ("C", "D")],
        train_labels=[1, 1],
        test_pairs=[("A", "C"), ("B", "D")],
        test_labels=[0, 0],
    )
    receipt = audit_signor_ggi_leakage(
        dataset=dataset,
        signor_path=signor,
        corpus_path=corpus,
        output_path=tmp_path / "receipt.json",
    )
    positive = receipt["sections"]["train"]["1"]
    assert positive["source_signor_overlap"] == 2
    assert positive["text_exposed_signor_overlap"] == 1
