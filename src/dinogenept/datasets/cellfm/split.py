"""Native simulation split, audited against CellFM's vendored GEARS splitter.

Reference revision bfed59c0e34103231165d69b97927ecc888d623c, data_utils.py
DataSplitter.split_data/get_simulation_split and pertdata.py prepare_split.
Reproduce MT19937 choice and reseeding, not numpy.default_rng. No global RNG
mutation. Custom prespecified test genes and other split modes are out of scope.
"""

import numpy as np


def targets(condition: str) -> tuple[str, ...]:
    if condition == "ctrl":
        return ()
    pieces = condition.split("+")
    if len(pieces) != 2 or any(not value for value in pieces):
        raise ValueError(f"Expected CellFM/GEARS A+ctrl or A+B condition: {condition}")
    genes = tuple(x for x in pieces if x != "ctrl")
    if not genes or len(set(genes)) != len(genes):
        raise ValueError("Empty or repeated perturbation target")
    return genes


def _partition(conditions, fraction, combo_fraction, seed):
    parsed = {c: targets(c) for c in conditions}
    gene_universe = np.unique([gene for genes in parsed.values() for gene in genes])
    selected = np.random.RandomState(seed).choice(gene_universe, int(len(gene_universe) * fraction), replace=False)
    seen = set(selected)
    singles = [c for c in conditions if len(parsed[c]) == 1]
    combos = [c for c in conditions if len(parsed[c]) == 2]
    train_single = [c for c in singles if parsed[c][0] in seen]
    unseen_single = [c for c in singles if parsed[c][0] not in seen]
    seen1 = [c for c in combos if sum(g in seen for g in parsed[c]) == 1]
    seen0 = [c for c in combos if not any(g in seen for g in parsed[c])]
    seen2_candidates = np.array(sorted(c for c in combos if all(g in seen for g in parsed[c])), dtype=str)
    train_combo = (
        np.random.RandomState(seed)
        .choice(seen2_candidates, int(len(seen2_candidates) * combo_fraction), replace=False)
        .tolist()
    )
    seen2 = np.setdiff1d(seen2_candidates, train_combo).tolist()
    train = train_single + train_combo
    test = seen1 + seen2 + unseen_single + seen0
    if len(train) + len(test) != len(conditions) or set(train) & set(test):
        raise ValueError("Simulation partition lost or duplicated a condition")
    return (
        train,
        test,
        {"combo_seen0": seen0, "combo_seen1": seen1, "combo_seen2": seen2, "unseen_single": unseen_single},
    )


def simulation_split(conditions, *, seed=3, train_gene_fraction=0.75, combo_train_fraction=0.75):
    """Return ordered condition membership and exact upstream subgroup lists.

    Input order is first observation occurrence, as in adata.obs.unique().
    Validation is a second simulation split with both fractions 0.9 and the
    SAME seed, not an independent random cell split. Controls belong to train.
    """
    if not 0 < train_gene_fraction < 1 or not 0 < combo_train_fraction <= 1:
        raise ValueError("Invalid training fractions")
    ordered = list(dict.fromkeys(conditions))
    if "ctrl" not in ordered:
        raise ValueError("CellFM perturbation data must contain control cells")
    perts = [x for x in ordered if x != "ctrl"]
    train, test, test_groups = _partition(perts, train_gene_fraction, combo_train_fraction, seed)
    train, validation, validation_groups = _partition(train, 0.9, 0.9, seed)
    memberships = {"train": set(train) | {"ctrl"}, "val": set(validation), "test": set(test)}
    if set.union(*memberships.values()) != set(ordered):
        raise ValueError("Unassigned perturbation condition")
    return {
        "conditions": {split: [c for c in ordered if c in members] for split, members in memberships.items()},
        "subgroups": {"test_subgroup": test_groups, "val_subgroup": validation_groups},
        "seed": seed,
        "train_gene_fraction": train_gene_fraction,
        "combo_train_fraction": combo_train_fraction,
        "validation_gene_fraction": 0.9,
        "validation_combo_fraction": 0.9,
        "protocol": "cellfm.gears.simulation.bfed59c",
    }
