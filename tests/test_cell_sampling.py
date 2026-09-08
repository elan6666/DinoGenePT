import numpy as np
import pytest

from dinogenept.cell.sampling import CropConfig, collate_crops, epoch_batches, sample_crops


def test_batch_half_globals_share_twenty_percent_reconstruction_mask():
    pytest.importorskip("torch")
    rows = [sample_crops(np.arange(1, 101), np.ones(100), 100,
                         cell_id=str(i), epoch=3,
                         config=CropConfig(global_scale=(1., 1.))) for i in range(8)]
    batch = collate_crops(rows)
    repeat = collate_crops(rows)
    masked = []
    for index, view in enumerate(batch["views"]):
        assert (view["targets"] == view["hidden"]).all()
        assert not (view["hidden"] & ~view["valid"]).any()
        assert (view["hidden"] == repeat["views"][index]["hidden"]).all()
        if index < 2:
            counts = view["hidden"].sum(-1)
            assert ((counts == 0) | (counts == 20)).all()
            masked.append(counts > 0)
        else:
            assert not view["hidden"].any()
    assert sum(int(x.sum()) for x in masked) == 8
    assert (masked[0] == masked[1]).any()  # not forced to exactly one per cell
    assert all(row["views"][0]["targets"].sum() == 20 for row in rows)


def test_continuous_crops_preserve_full_library_and_seed():
    ids = np.arange(1, 41)
    counts = np.arange(40, dtype=float)
    config = CropConfig(cap=20)
    args = dict(cell_id="study:donor:cell", epoch=1, config=config)
    first = sample_crops(ids, counts, 1000, **args)
    second = sample_crops(ids, counts, 1000, **args)
    assert len(first["capped_gene_ids"]) == 20 and 1 not in first["capped_gene_ids"]
    assert len(first["views"]) == 4
    for i, (a, b) in enumerate(zip(first["views"], second["views"], strict=True)):
        for name in ("gene_ids", "expression", "targets", "hidden"):
            np.testing.assert_array_equal(a[name], b[name])
        assert np.all(np.diff(a["gene_ids"]) > 0)
        expected = np.log1p(10 * counts[a["gene_ids"] - 1])
        np.testing.assert_allclose(a["expression"], expected, atol=1e-6)
        assert set(a["gene_ids"]) <= set(first["capped_gene_ids"])
        assert np.all(~a["hidden"] | a["targets"])
        if i < 2:
            assert 8 <= len(a["gene_ids"]) <= 20
            assert a["hidden"].any()
        else:
            assert 1 <= len(a["gene_ids"]) <= 8
            assert not a["targets"].any()


@pytest.mark.parametrize("count", [17, 31, 32, 33, 65])
def test_epoch_visits_every_cell_once_across_ranks(count):
    ranks = [list(epoch_batches(count, 8, 2, rank, seed=42, epoch=1)) for rank in range(2)]
    assert len(ranks[0]) == len(ranks[1])
    all_ids = [i for rank in ranks for batch in rank for i in batch]
    assert sorted(all_ids) == list(range(count))
    assert all(2 <= len(batch) <= 8 for rank in ranks for batch in rank)
    assert ranks[0] == list(epoch_batches(count, 8, 2, 0, seed=42, epoch=1))


def test_bad_raw_values_and_insufficient_batch_are_errors():
    with pytest.raises(ValueError):
        sample_crops([1, 2], [2, 3], 4, cell_id="a", epoch=0, config=CropConfig())
    with pytest.raises(ValueError):
        sample_crops([1, 2], [0, 0], 10, cell_id="a", epoch=0, config=CropConfig())
    with pytest.raises(ValueError):
        list(epoch_batches(5, 2, 2, 0, seed=42, epoch=0))
