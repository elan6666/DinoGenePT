import numpy as np
import pytest

from dinogenept.datasets.populations import PopulationIndex, continuous_view


def fixture():
    conditions = ["ctrl"] * 4 + ["A+ctrl"] * 17 + ["B+ctrl"] * 3 + ["C+ctrl"] * 2
    return PopulationIndex(
        [f"row-{i}" for i in range(26)],
        conditions,
        ["K562"] * 26,
        {"train": ["ctrl", "A+ctrl"], "val": ["B+ctrl"], "test": ["C+ctrl"]},
    )


def test_each_epoch_visits_every_train_post_once_and_never_heldout_teacher():
    index = fixture()
    for epoch in (0, 1):
        bags = index.training_bags(epoch=epoch)
        assert sorted(i for bag in bags for i in bag.primary_post) == list(range(4, 21))
        assert bags == index.training_bags(epoch=epoch)
        for bag in bags:
            assert 1 <= len(bag.primary_post) <= 8
            assert len(bag.controls) == len(bag.teacher_post) == len(bag.observed_post) == 8
            assert set(bag.controls) <= set(range(4))
            assert set(bag.teacher_post + bag.observed_post) <= set(range(4, 21))
            assert not set(bag.teacher_post) & set(bag.observed_post)
    assert index.training_bags(epoch=0) != index.training_bags(epoch=1)


def test_donor_matching_singletons_and_exact_primary_coverage():
    index = PopulationIndex(
        [str(i) for i in range(6)], ["ctrl", "A", "A", "A", "A", "A"], ["T"] * 6,
        {"train": ["ctrl", "A"], "val": [], "test": []},
        {"donor": ["d0", "d1", "d1", "d2", "d2", "d3"]},
    )
    bags = index.training_bags(epoch=0)
    assert sorted(i for b in bags for i in b.primary_post) == list(range(1, 6))
    for b in bags:
        rows = b.primary_post + b.teacher_post + b.observed_post
        assert len(set(index.pairing_fields["donor"][list(rows)])) == 1
        assert not set(b.teacher_post) & set(b.observed_post)
        if 5 in b.primary_post:
            assert b.observed_post == ()


def test_evaluation_controls_are_frozen_and_truth_not_used_as_inputs():
    index = fixture()
    group = index.evaluation_groups("test")[0]
    assert group == index.evaluation_groups("test")[0]
    assert len(group["control_rows"]) == 300
    assert set(group["control_rows"]) <= set(range(4))
    assert group["truth_rows"] == (24, 25)
    with pytest.raises(ValueError):
        index.evaluation_groups("train")


def test_uniform_continuous_view_keeps_real_zeros_and_gene_order():
    axis = np.array([4, 1, 3, 2])
    counts = np.array([[0, 0, 1.2345, 0], [0, 2.345, 0, 0]])
    view = continuous_view(axis, counts, ["a", "b"], cap=4)
    np.testing.assert_array_equal(view["gene_ids"], [[1, 2, 3, 4]] * 2)
    np.testing.assert_allclose(view["expression"], counts[:, [1, 3, 2, 0]])
    assert view["valid"].all() and not view["hidden"].any()
    small = continuous_view(axis, np.zeros_like(counts), ["a", "b"], cap=2, mask_fraction=0.2)
    assert small["valid"].all() and small["hidden"].sum() == 2


def test_population_missing_context_control_and_overlapping_splits_fail():
    with pytest.raises(ValueError, match="overlap"):
        PopulationIndex(
            ["a", "b"], ["ctrl", "A+ctrl"], ["K562"] * 2, {"train": ["ctrl", "A+ctrl"], "val": ["A+ctrl"], "test": []}
        )
    with pytest.raises(ValueError, match="matched control"):
        PopulationIndex(
            ["a", "b"], ["ctrl", "A+ctrl"], ["K562", "RPE1"], {"train": ["ctrl"], "val": ["A+ctrl"], "test": []}
        )
