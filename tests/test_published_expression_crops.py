import numpy as np
import pytest

from dinogenept.cell.sampling import CropConfig, sample_crops


def test_published_values_are_not_renormalized():
    ids = np.arange(1, 11)
    values = np.arange(1, 11, dtype=float) / 3
    result = sample_crops(ids, values, None, cell_id="row0", epoch=0, config=CropConfig(),
                          expression_scale="genecompass_published_continuous")
    for view in result["views"]:
        np.testing.assert_array_equal(view["expression"], values[view["gene_ids"] - 1].astype(np.float32))


def test_continuous_rejects_fake_library_size():
    with pytest.raises(ValueError, match="fabricated"):
        sample_crops([1, 2], [0.3, 0.5], 0.8, cell_id="row0", epoch=0, config=CropConfig(),
                     expression_scale="genecompass_published_continuous")


def test_unknown_scale_fails_closed():
    with pytest.raises(ValueError, match="Unknown expression scale"):
        sample_crops([1, 2], [0.3, 0.5], None, cell_id="row0", epoch=0, config=CropConfig(),
                     expression_scale="bins")
