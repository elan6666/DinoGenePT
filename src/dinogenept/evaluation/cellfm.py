"""Model-independent CellFM evaluation state with GraD-Pert-aligned methods.

Reference read at GraD-Pert 276d7ba: evaluation/state.py and data/controls.py.
No GraD-Pert runtime import. Reduced CellFM axes/splits are NOT its canonical
datasets. DE and train+validation reference are evaluator-only, never features.
"""

import hashlib
import json
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dinogenept.datasets.cellfm.split import targets
from dinogenept.provenance import atomic_write_json, digest_file


def control_draws(dataset_id, split, condition, truth_contexts, pools, *, seed=20260824):
    """Source-matched context-resampling followed by 300 replacement row draws."""
    if split not in {"val", "test"} or not truth_contexts or not dataset_id or not condition:
        raise ValueError("Invalid evaluation draw identity")
    if any(context not in pools or not len(pools[context]) for context in truth_contexts):
        raise ValueError("Evaluation context has no compatible controls")
    if any(len(np.unique(pool)) != len(pool) for pool in pools.values()):
        raise ValueError("Source control pool contains duplicate rows")
    key = f"{seed}::{dataset_id}::{split}::{condition}".encode()
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(hashlib.sha256(key).digest()[:16], "big")))
    contexts = rng.choice(truth_contexts, size=300, replace=True).tolist()
    return [int(rng.choice(pools[context])) for context in contexts], contexts


def select_de(ranked, symbols, condition, perturbed_mean, control_mean):
    """Non-dropout -> top20 -> target exclusion (no refill), as upstream."""
    index = {gene: i for i, gene in enumerate(symbols)}
    non_dropout = (perturbed_mean != 0) | ((perturbed_mean == 0) & (control_mean == 0))
    eligible = [index[gene] for gene in ranked if gene in index and non_dropout[index[gene]]][:20]
    target_indices = {index[gene] for gene in targets(condition) if gene in index}
    selected = sorted(set(eligible) - target_indices)
    if not selected:
        raise ValueError(f"No DE genes after target exclusion: {condition}")
    return selected


def prepare_evaluation(data, path, *, dataset_id, seed=20260824):
    """Use Scanpy t-test exactly as reference; never use whole-data HVG fitting."""
    import anndata as ad
    import pandas as pd
    import scanpy as sc

    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserve an existing frozen evaluation state")
    matrix, index = data._matrix, data.index
    conditions, contexts = index.conditions, index.contexts
    all_control = np.flatnonzero(conditions == "ctrl")
    evaluation_conditions = [*data.split_conditions["val"], *data.split_conditions["test"]]
    rows_by_condition = {c: np.flatnonzero(conditions == c) for c in set(conditions)}
    rankable = [c for c in evaluation_conditions if len(rows_by_condition[c]) >= 2]
    ranked = {}
    if rankable:
        selected = np.flatnonzero((conditions == "ctrl") | np.isin(conditions, rankable))
        ranking = ad.AnnData(
            X=matrix[selected].copy(),
            obs=pd.DataFrame({"condition": pd.Categorical(conditions[selected])}, index=index.row_ids[selected]),
            var=pd.DataFrame(index=data.symbols),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
            sc.tl.rank_genes_groups(
                ranking,
                groupby="condition",
                reference="ctrl",
                rankby_abs=True,
                n_genes=len(data.axis),
                method="t-test",
            )
        names = ranking.uns["rank_genes_groups"]["names"]
        ranked = {c: [str(g) for g in names[c].tolist()] for c in names.dtype.names}
        del ranking

    def mean(rows):
        # Match reference state construction: sparse mean -> float32, then
        # equal-condition sum in float64. Metric correlations later use float64.
        return np.asarray(matrix[rows].mean(axis=0), dtype=np.float32).ravel()

    control_mean = mean(all_control)
    reference_conditions = [c for c in [*data.split_conditions["train"], *data.split_conditions["val"]] if c != "ctrl"]
    reference = np.zeros(len(data.axis), dtype=np.float64)
    for condition in reference_conditions:
        reference += mean(rows_by_condition[condition])
    reference = (reference / len(reference_conditions)).astype(np.float32)
    pools = {context: index.groups[("ctrl", context)] for context in sorted(set(contexts[all_control]))}
    rows = {}
    for condition in evaluation_conditions:
        truth = rows_by_condition[condition]
        split = "val" if condition in index.splits["val"] else "test"
        selected_controls, selected_contexts = control_draws(
            dataset_id, split, condition, contexts[truth].tolist(), pools, seed=seed
        )
        compatible = np.concatenate([pools[context] for context in sorted(set(contexts[truth]))])
        reason = f"insufficient_truth_cells_for_t_test:n={len(truth)}" if len(truth) < 2 else None
        de = [] if reason else select_de(ranked[condition], data.symbols, condition, mean(truth), control_mean)
        rows[condition] = {
            "split": split,
            "truth_row_ids": index.row_ids[truth].tolist(),
            "control_row_ids": index.row_ids[selected_controls].tolist(),
            "control_contexts": selected_contexts,
            "metric_control_mean": mean(compatible).tolist(),
            "de_indices": de,
            "top_de_indices": de,
            "de_unavailable_reason": reason,
        }
    state = {
        "schema": "dinogenept.cellfm.evaluation.v1",
        "identity": data.identity,
        "dataset_id": dataset_id,
        "evaluation_seed": seed,
        "rng": "PCG64",
        "sample_with_replacement": True,
        "de_method": "scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets",
        "reference_commit": "276d7baac57c6a4130b5e43563cc7eccd8c8ff7e",
        "scanpy_version": version("scanpy"),
        "scope": "GraD-Pert_methods_on_CellFM_released_reduced_axis_not_canonical_data_parity",
        "reference_conditions": reference_conditions,
        "systema_reference": reference.tolist(),
        "conditions": rows,
    }
    atomic_write_json(path, state)
    return state


def load_evaluation(data, path, *, sha256):
    path = Path(path)
    if digest_file(path) != sha256:
        raise ValueError("Frozen evaluation checksum differs")
    state = json.loads(path.read_text())
    if state.get("schema") != "dinogenept.cellfm.evaluation.v1" or state.get("identity") != data.identity:
        raise ValueError("Evaluation state belongs to different data/axis/split")
    if (
        state.get("evaluation_seed") != 20260824
        or state.get("rng") != "PCG64"
        or state.get("sample_with_replacement") is not True
        or state.get("de_method") != "scanpy_t_test_rankby_abs_non_dropout_top20_exclude_targets"
    ):
        raise ValueError("Evaluation control/DE protocol differs from the frozen method")
    expected_reference = [c for c in [*data.split_conditions["train"], *data.split_conditions["val"]] if c != "ctrl"]
    if state.get("reference_conditions") != expected_reference:
        raise ValueError("Systema reference must use exactly train+validation noncontrol conditions")
    expected = data.index.splits["val"] | data.index.splits["test"]
    if set(state["conditions"]) != expected:
        raise ValueError("Evaluation conditions differ from split")
    lookup = {row: i for i, row in enumerate(data.index.row_ids)}
    reference = np.asarray(state["systema_reference"], dtype=np.float32)
    if reference.shape != (len(data.axis),) or not np.isfinite(reference).all():
        raise ValueError("Invalid Systema reference axis/values")
    for condition, row in state["conditions"].items():
        split = "val" if condition in data.index.splits["val"] else "test"
        truth = np.flatnonzero(data.index.conditions == condition)
        if row["split"] != split or row["truth_row_ids"] != data.index.row_ids[truth].tolist():
            raise ValueError("Evaluation truth membership/order differs")
        controls = np.asarray([lookup[r] for r in row["control_row_ids"]])
        if len(controls) != 300 or not np.all(data.index.conditions[controls] == "ctrl"):
            raise ValueError("Evaluation controls must be exactly 300 real control draws")
        if data.index.contexts[controls].tolist() != row["control_contexts"]:
            raise ValueError("Control context order differs")
        if set(row["control_contexts"]) - set(data.index.contexts[truth]):
            raise ValueError("Controls mismatch truth context")
        truth_contexts = data.index.contexts[truth].tolist()
        pools = {context: data.index.groups[("ctrl", context)] for context in sorted(set(truth_contexts))}
        expected_controls, expected_contexts = control_draws(
            state["dataset_id"], split, condition, truth_contexts, pools, seed=state["evaluation_seed"]
        )
        if controls.tolist() != expected_controls or row["control_contexts"] != expected_contexts:
            raise ValueError("Frozen ordered controls differ from the prescribed RNG draw")
        pool = np.asarray(row["metric_control_mean"], dtype=np.float32)
        if pool.shape != reference.shape or not np.isfinite(pool).all():
            raise ValueError("Invalid metric control axis/values")
        for key in ("de_indices", "top_de_indices"):
            indices = row[key]
            if indices != sorted(set(indices)) or any(
                type(i) is not int or i < 0 or i >= len(data.axis) for i in indices
            ):
                raise ValueError("DE indices are not unique valid expression positions")
            if (not indices) != (row["de_unavailable_reason"] is not None):
                raise ValueError("Missing DE must have an explicit reason")
            excluded = {data.symbols.index(gene) for gene in targets(condition) if gene in data.symbols}
            if len(indices) > 20 or excluded & set(indices):
                raise ValueError("DE list violates top20/target-exclusion policy")
        if row["de_indices"] != row["top_de_indices"]:
            raise ValueError("This reference defines the same DE and TopDE index sets")
    return state
