"""Evaluation methods aligned with selected GenePT notebook experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .tasks import GGIDataset, PropertyTask


def _matrix(genes: list[str], vectors: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, gene in enumerate(genes) if gene in vectors]
    if not indices:
        raise ValueError("no task genes have embeddings")
    return np.stack([vectors[genes[i]] for i in indices]), np.asarray(indices)


def evaluate_property_task(
    task: PropertyTask,
    vectors: Mapping[str, np.ndarray],
    *,
    folds: int = 5,
    random_state: int = 42,
) -> list[dict[str, object]]:
    x, indices = _matrix(task.genes, vectors)
    y = np.asarray(task.labels)[indices]
    if len(np.unique(y)) != 2:
        raise ValueError("property task requires both classes after coverage filtering")
    minority = int(np.bincount(y).min())
    if minority < folds:
        raise ValueError(f"each class requires at least {folds} examples")
    split = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    model_factories = {
        "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=random_state),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=500, random_state=random_state
        ),
    }
    results = []
    for model_name, factory in model_factories.items():
        for fold, (train, test) in enumerate(split.split(x, y), start=1):
            model = factory()
            model.fit(x[train], y[train])
            score = model.predict_proba(x[test])[:, 1]
            prediction = (score >= 0.5).astype(int)
            results.append(
                {
                    "task": task.name,
                    "model": model_name,
                    "fold": fold,
                    "n": len(y),
                    "coverage": len(y) / len(task.genes),
                    "accuracy": float(accuracy_score(y[test], prediction)),
                    "auroc": float(roc_auc_score(y[test], score)),
                    "average_precision": float(average_precision_score(y[test], score)),
                }
            )
    return results


def evaluate_property_task_repeated(
    task: PropertyTask,
    vectors: Mapping[str, np.ndarray],
    *,
    folds: int = 5,
    seeds: tuple[int, ...] = tuple(range(42, 52)),
) -> list[dict[str, object]]:
    rows = []
    for seed in seeds:
        for row in evaluate_property_task(task, vectors, folds=folds, random_state=seed):
            rows.append({"random_state": seed, **row})
    return rows


def _pair_matrix(
    pairs: list[tuple[str, str]], vectors: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    indices = [i for i, (left, right) in enumerate(pairs) if left in vectors and right in vectors]
    if not indices:
        raise ValueError("no GGI pairs have embeddings for both genes")
    matrix = np.stack([vectors[pairs[i][0]] + vectors[pairs[i][1]] for i in indices])
    return matrix, np.asarray(indices)


def evaluate_ggi(
    dataset: GGIDataset,
    vectors: Mapping[str, np.ndarray],
    *,
    random_state: int = 42,
) -> dict[str, object]:
    train_x, train_indices = _pair_matrix(dataset.train_pairs, vectors)
    test_x, test_indices = _pair_matrix(dataset.test_pairs, vectors)
    train_y = np.asarray(dataset.train_labels)[train_indices]
    test_y = np.asarray(dataset.test_labels)[test_indices]
    if len(np.unique(train_y)) != 2 or len(np.unique(test_y)) != 2:
        raise ValueError("GGI train and test sets require both classes after coverage filtering")
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(train_x, train_y)
    score = model.predict_proba(test_x)[:, 1]
    prediction = (score >= 0.5).astype(int)
    return {
        "task": "gene_gene_interaction",
        "model": "logistic_regression",
        "pair_operator": "sum",
        "train_n": len(train_y),
        "test_n": len(test_y),
        "train_coverage": len(train_y) / len(dataset.train_pairs),
        "test_coverage": len(test_y) / len(dataset.test_pairs),
        "accuracy": float(accuracy_score(test_y, prediction)),
        "auroc": float(roc_auc_score(test_y, score)),
        "average_precision": float(average_precision_score(test_y, score)),
    }
