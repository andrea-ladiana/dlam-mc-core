import pytest

from dlam_mc_core.generation import generate_layer
from dlam_mc_core.types import SimParams


@pytest.mark.parametrize(
    "kwargs",
    [
        {"N1": 0},
        {"M1": 0},
        {"r1": 0.0},
        {"t1": -0.1},
        {"beta": 0.0},
        {"steps": 0},
        {"cue_layer": 4},
        {"mu0": 40},
        {"init_mode": "invalid"},
        {"init_noise": 1.5},
    ],
)
def test_simparams_validation_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SimParams(**kwargs)


def test_generate_layer_validation_rejects_bad_arguments() -> None:
    rng = __import__("numpy").random.default_rng(0)

    with pytest.raises(ValueError, match="strictly positive"):
        generate_layer(K=0, N=10, M=2, r=0.7, rng=rng)

    with pytest.raises(ValueError, match=r"in \(0, 1\]"):
        generate_layer(K=2, N=10, M=2, r=0.0, rng=rng)
