"""Condition/context-matched unpaired bags with strict split separation.

Population bagging is our documented adaptation, not a claim of true matched
single cells. Each training post cell is a primary target once per epoch.
Teacher/observed bags are disjoint draws from matched TRAIN groups. Evaluation control
IDs are fixed independently of all post-expression values.
"""

from dataclasses import dataclass

import numpy as np

from dinogenept.cell.sampling import cell_rng


@dataclass(frozen=True)
class TrainingBag:
    condition: str
    context: str
    key: str
    primary_post: tuple[int, ...]
    controls: tuple[int, ...]
    observed_post: tuple[int, ...]
    teacher_post: tuple[int, ...]


class PopulationIndex:
    def __init__(self, row_ids, conditions, contexts, splits, pairing_fields=None):
        self.row_ids = np.asarray(row_ids, dtype=str)
        self.conditions = np.asarray(conditions, dtype=str)
        self.contexts = np.asarray(contexts, dtype=str)
        self.pairing_fields = {k: np.asarray(v, dtype=str) for k, v in (pairing_fields or {}).items()}
        if any(v.shape != self.row_ids.shape for v in self.pairing_fields.values()):
            raise ValueError("Pairing metadata axes differ")
        if (
            self.row_ids.ndim != 1
            or self.conditions.shape != self.row_ids.shape
            or self.contexts.shape != self.row_ids.shape
        ):
            raise ValueError("Population row metadata axes differ")
        if not len(self.row_ids) or len(np.unique(self.row_ids)) != len(self.row_ids):
            raise ValueError("Population row IDs must be unique and nonempty")
        if any(not x for x in self.row_ids) or any(not x for x in self.contexts):
            raise ValueError("Empty row/context identity")
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("Expected explicit train/val/test condition membership")
        self.splits = {name: set(values) for name, values in splits.items()}
        members = [value for values in splits.values() for value in values]
        if len(members) != len(set(members)) or set(members) != set(self.conditions):
            raise ValueError("Split conditions overlap or do not exactly cover observations")
        if "ctrl" not in self.splits["train"]:
            raise ValueError("Control pool must be assigned to train")
        self.groups = {}
        for i, (condition, context) in enumerate(zip(self.conditions, self.contexts, strict=True)):
            self.groups.setdefault((condition, context), []).append(i)
        self.groups = {key: np.asarray(value, dtype=np.int64) for key, value in self.groups.items()}
        self.training_groups = {}
        for (condition, context), rows in self.groups.items():
            for row in rows:
                signature = tuple(values[row] for values in self.pairing_fields.values())
                self.training_groups.setdefault((condition, context, signature), []).append(int(row))
        if any(("ctrl", context) not in self.groups for condition, context in self.groups if condition != "ctrl"):
            raise ValueError("Every perturbation context requires a matched control pool")

    @staticmethod
    def _draw(rng, values, size):
        return tuple(int(i) for i in rng.choice(values, size=size, replace=len(values) < size))

    def training_bags(self, *, epoch, seed=42, bag_size=8):
        if epoch < 0 or bag_size < 1:
            raise ValueError("Invalid bag epoch/size")
        bags = []
        for (condition, context, signature), rows in sorted(self.training_groups.items()):
            if condition == "ctrl" or condition not in self.splits["train"]:
                continue
            group_id = repr((condition, context, signature))
            rng = cell_rng(seed, epoch, group_id)
            shuffled = rng.permutation(rows)
            count = (len(rows) + bag_size - 1) // bag_size
            for j, primary in enumerate(np.array_split(shuffled, count)):
                candidates = rng.permutation(rows)
                n_teacher = min(bag_size, max(1, len(candidates) // 2))
                teacher = tuple(int(i) for i in candidates[:n_teacher])
                observed = tuple(int(i) for i in candidates[n_teacher:n_teacher + bag_size])
                bags.append(
                    TrainingBag(
                        condition=condition,
                        context=context,
                        key=f"{group_id}\x1f{j}",
                        primary_post=tuple(int(i) for i in primary),
                        controls=self._draw(rng, self.groups[("ctrl", context)], bag_size),
                        observed_post=observed,
                        teacher_post=teacher,
                    )
                )
        if not bags:
            raise ValueError("No train perturbation observations")
        order = cell_rng(seed, epoch, "bag-order").permutation(len(bags))
        return [bags[i] for i in order]

    def evaluation_groups(self, split, *, seed=42, controls=300):
        if split not in {"val", "test"} or controls != 300:
            raise ValueError("Evaluation uses val/test with exactly 300 ordered control draws")
        result = []
        for (condition, context), rows in sorted(self.groups.items()):
            if condition not in self.splits[split]:
                continue
            rng = cell_rng(seed, 0, f"eval:{split}:{condition}\x1f{context}")
            result.append(
                {
                    "condition": condition,
                    "context": context,
                    "truth_rows": tuple(rows.tolist()),
                    "control_rows": self._draw(rng, self.groups[("ctrl", context)], controls),
                }
            )
        return result


def continuous_view(axis, expression, row_ids, *, seed=42, epoch=0, role="control", cap=2048, mask_fraction=0.0):
    """Stage-2 capped uniform gene sampling: true zeros remain eligible.

    `axis` contains only measured genes, already mapped to pretrained IDs. Values
    are continuous normalized expression; do not normalize or bin them again.
    Masking is an observed-view augmentation, not an added reconstruction loss.
    """
    axis, expression = np.asarray(axis), np.asarray(expression, dtype=np.float32)
    if (
        axis.ndim != 1
        or not np.issubdtype(axis.dtype, np.integer)
        or len(np.unique(axis)) != len(axis)
        or np.any(axis < 1)
    ):
        raise ValueError("Measured gene axis must contain unique positive pretrained IDs")
    if expression.shape != (len(row_ids), len(axis)) or not np.isfinite(expression).all() or np.any(expression < 0):
        raise ValueError("Continuous expression axes/values invalid")
    if cap < 1 or not len(axis) or not 0 <= mask_fraction < 1:
        raise ValueError("Invalid view cap/masking fraction")
    size = min(cap, len(axis))
    ids, values, masks = [], [], []
    for i, row_id in enumerate(row_ids):
        rng = cell_rng(seed, epoch, f"{role}:{row_id}")
        selected = rng.choice(len(axis), size=size, replace=False) if size < len(axis) else np.arange(size)
        selected = selected[np.argsort(axis[selected])]
        hidden = np.zeros(size, dtype=bool)
        if mask_fraction:
            hidden[rng.choice(size, size=max(1, int(size * mask_fraction)), replace=False)] = True
        ids.append(axis[selected])
        values.append(expression[i, selected])
        masks.append(hidden)
    return {
        "gene_ids": np.stack(ids),
        "expression": np.stack(values),
        "valid": np.ones((len(row_ids), size), dtype=bool),
        "hidden": np.stack(masks),
    }
