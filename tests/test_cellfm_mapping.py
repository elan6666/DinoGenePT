import pytest

from dinogenept.datasets.cellfm.mapping import map_axis


def test_mapping_preserves_axis_order_and_uses_nonzero_native_ids():
    records = [{"feature_id": "E1", "feature_name": "A"}, {"feature_id": "E2", "feature_name": "B"}]
    result = map_axis(["B", "A"], ["A"], ["E1", "E2"], records)
    assert result["model_gene_ids"] == [2, 1]
    assert result["target_model_ids"] == {"A": 1}


def test_unknown_and_ambiguous_symbols_fail_without_dropping_genes():
    records = [{"feature_id": "E1", "feature_name": "A"}, {"feature_id": "E2", "feature_name": "A"}]
    with pytest.raises(ValueError, match="ambiguous"):
        map_axis(["A"], [], ["E1", "E2"], records)
    with pytest.raises(ValueError, match="Missing"):
        map_axis(["B"], [], ["E1", "E2"], records)
