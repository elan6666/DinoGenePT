import numpy as np

from dinogenept.evaluation.evaluator import PerturbationEvaluator


def test_common_evaluator_reports_condition_macro_and_availability_strata():
    evaluator = PerturbationEvaluator(
        {"bootstrap_resamples": 0, "energy_distance": False, "bootstrap_seed": 1}
    )
    control = np.zeros((3, 4), dtype=np.float32)
    predictions = {
        "A+ctrl": np.ones((3, 4), dtype=np.float32),
        "B+ctrl": np.full((3, 4), 2.0, dtype=np.float32),
    }
    truths = {key: value.copy() for key, value in predictions.items()}
    controls = {key: control for key in predictions}
    result = evaluator.evaluate(
        predictions=predictions,
        truths=truths,
        controls=controls,
        top_de_indices={"A+ctrl": (0, 1), "B+ctrl": (0, 1)},
        strata={
            "optional_source_count": {"A+ctrl": "0", "B+ctrl": "1"}
        },
    )
    assert result["statistical_unit"] == "perturbation_condition"
    assert result["summary"]["mse"]["macro_mean"] == 0.0
    assert set(result["stratified"]["optional_source_count"]) == {"0", "1"}
