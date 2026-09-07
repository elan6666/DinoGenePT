import ast
from pathlib import Path

import numpy as np
import pytest

from dinogenept.datasets.cellfm.split import simulation_split
from dinogenept.provenance import digest_file


def conditions():
    genes = [f"G{i:02}" for i in range(20)]
    return [
        "ctrl",
        *[f"{g}+ctrl" for g in genes],
        *[f"{left}+{right}" for i, left in enumerate(genes) for right in genes[i + 1 :]],
    ]


def test_split_is_condition_disjoint_and_does_not_mutate_rng():
    np.random.seed(10)
    before = np.random.get_state()
    result = simulation_split(conditions())
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    splits = [set(x) for x in result["conditions"].values()]
    assert not any(splits[i] & splits[j] for i in range(3) for j in range(i + 1, 3))
    assert set.union(*splits) == set(conditions())
    assert "ctrl" in result["conditions"]["train"]
    assert result == simulation_split(conditions())


def test_against_pinned_upstream_splitter_when_reference_is_available():
    # Test-only source execution. Production never imports upstream packages.
    root = Path(__file__).resolve().parents[1] / ".runtime/references"
    source = root / "cellfm-data_utils-bfed59c.py"
    helpers = root / "cellfm-utils-bfed59c.py"
    if not source.exists() or not helpers.exists():
        pytest.skip("Pinned reference source not downloaded in this environment")
    assert digest_file(source) == "776b41af9c8d49131d103ef54c8e0bea892a8313137e9c191fd8e74a41c05b58"
    assert digest_file(helpers) == "d24d35a50741b011e5e8078d2666aa0d1dfea535a98101691d523ab00e419a9a"
    pd = pytest.importorskip("pandas")
    nodes = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "DataSplitter"
    ]
    nodes += [
        node
        for node in ast.parse(helpers.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"parse_any_pert", "parse_single_pert", "parse_combo_pert"}
    ]
    scope = {"np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), scope)

    class Cells:
        def __init__(self, rows):
            self.obs = pd.DataFrame({"condition": rows})

    for rows in (conditions(), [x for x in conditions() if x == "ctrl" or x.endswith("+ctrl")]):
        for seed in (1, 3, 42):
            original, groups = scope["DataSplitter"](Cells(rows), split_type="simulation").split_data(seed=seed)
            native = simulation_split(rows, seed=seed)
            expected = {
                split: original.obs.loc[original.obs.split == split, "condition"].unique().tolist()
                for split in ("train", "val", "test")
            }
            assert native["conditions"] == expected
            assert native["subgroups"] == groups
