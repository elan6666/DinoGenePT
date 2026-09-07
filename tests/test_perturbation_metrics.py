"""Synthetic contract checks: no data access and no upstream model imports."""

import numpy as np
import pytest

from dinogenept.evaluation.metrics import METRICS, evaluate_condition, macro_average, pearson, systema_reference


def inputs():
    rng = np.random.default_rng(19)
    return dict(
        prediction=rng.normal(size=(300, 7)),
        input_control=rng.normal(size=(300, 7)),
        truth=rng.normal(size=(13, 7)),
        metric_control_mean=rng.normal(size=7),
        reference=rng.normal(size=7),
        de_indices=[1, 3, 4, 6],
        top_de_indices=[0, 2, 5],
    )


def test_distinct_references_and_unequal_populations():
    args = inputs()
    result = evaluate_condition(**args)
    pm, cm, tm = (args[k].mean(0) for k in ("prediction", "input_control", "truth"))
    refs = (cm, args["metric_control_mean"], args["reference"])
    axes = (slice(None), args["de_indices"], args["top_de_indices"])
    for key, ref, axis in zip(METRICS, refs, axes, strict=True):
        expected = np.corrcoef((pm - ref)[axis], (tm - ref)[axis])[0, 1]
        assert result[key].value == pytest.approx(expected, abs=1e-12)


def test_unavailable_and_macro_denominator():
    args = inputs()
    good = evaluate_condition(**args)
    args.update(de_indices=[], top_de_indices=[], de_unavailable_reason="missing")
    missing = evaluate_condition(**args)
    summary = macro_average({"a": good, "b": missing})["metrics"]
    assert summary[METRICS[0]]["finite_condition_count"] == 2
    assert summary[METRICS[1]]["finite_condition_count"] == 1
    assert summary[METRICS[1]]["total_condition_count"] == 2
    assert summary[METRICS[1]]["unavailable_reasons"] == ["de_unavailable:missing"]


@pytest.mark.parametrize("indices", [[1, 1], [1, 99], [1.1, 2.0], []])
def test_reject_invalid_indices(indices):
    args = inputs()
    args["de_indices"] = indices
    with pytest.raises(ValueError):
        evaluate_condition(**args)


def test_reject_299_controls():
    args = inputs()
    args["input_control"] = args["input_control"][:299]
    with pytest.raises(ValueError, match="300"):
        evaluate_condition(**args)


def test_undefined_pearson_and_equal_condition_reference():
    assert pearson(np.ones(4), np.arange(4)).reason == "constant_vector"
    assert pearson(np.array([np.nan, 1]), np.ones(2)).reason == "non_finite_input"
    assert pearson(np.ones(1), np.ones(1)).reason == "fewer_than_two_genes"
    actual = systema_reference({"a": np.zeros((1, 4)), "b": np.full((19, 4), 10)})
    np.testing.assert_allclose(actual, 5)
