import numpy as np

from dinogenept.priors import PriorStore


def _config(optional_coverage: float = 0.5):
    return {
        "priors": {
            "base": {"fixture": "deterministic", "dimension": 8, "coverage": 1.0},
            "optional": {
                "go": {
                    "fixture": "deterministic",
                    "dimension": 8,
                    "coverage": optional_coverage,
                    "source_only": True,
                }
            },
        }
    }


def test_optional_missingness_is_omission_not_a_fake_token():
    genes = tuple(f"G{index}" for index in range(40))
    store = PriorStore.from_config(_config(), genes)
    missing = next(gene for gene in genes if not store.source_available("go", (gene,)))
    vectors, token_mask, condition_mask = store.batch("go", ((missing,),))
    assert condition_mask.tolist() == [False]
    assert token_mask.tolist() == [[False]]
    assert np.count_nonzero(vectors) == 0


def test_combination_requires_every_target_and_controls_keep_availability():
    genes = tuple(f"G{index}" for index in range(40))
    store = PriorStore.from_config(_config(optional_coverage=1.0), genes)
    targets = ((genes[0], genes[1]),)
    original = store.batch("go", targets)
    shuffled = store.batch("go", targets, control="shuffled", seed=9)
    availability = store.batch("go", targets, control="availability_only")
    assert original[1].all() and shuffled[1].all() and availability[1].all()
    assert not np.allclose(original[0], shuffled[0])
    assert np.allclose(availability[0][0, 0], availability[0][0, 1])
