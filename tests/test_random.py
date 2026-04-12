import numpy as np
import pytest

from dlam_mc_core.random import init_disentangle, rademacher


def test_rademacher_binary_output_and_dtype() -> None:
    rng = np.random.default_rng(3)
    out = rademacher((16,), rng, dtype=np.int8)

    assert out.dtype == np.int8
    assert set(np.unique(out)).issubset({-1, 1})


def test_init_disentangle_returns_identical_cue_for_all_layers() -> None:
    rng = np.random.default_rng(0)
    xi1 = np.array([1, 1, -1, -1], dtype=np.int8)
    xi2 = np.array([1, -1, 1, -1], dtype=np.int8)
    xi3 = np.array([-1, 1, 1, -1], dtype=np.int8)

    s1, s2, s3 = init_disentangle(xi1, xi2, xi3, epsilon=0.0, rng=rng)

    assert np.array_equal(s1, s2)
    assert np.array_equal(s2, s3)
    assert set(np.unique(s1)).issubset({-1, 1})


@pytest.mark.parametrize(
    ("xi1", "xi2", "xi3", "epsilon", "match"),
    [
        (
            np.ones((2, 2), dtype=np.int8),
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=np.int8),
            0.0,
            "one-dimensional",
        ),
        (
            np.ones(3, dtype=np.int8),
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=np.int8),
            0.0,
            "requires N1 == N2 == N3",
        ),
        (
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=np.int8),
            np.ones(4, dtype=np.int8),
            1.2,
            r"in \[0, 1\]",
        ),
    ],
)
def test_init_disentangle_input_validation(
    xi1: np.ndarray,
    xi2: np.ndarray,
    xi3: np.ndarray,
    epsilon: float,
    match: str,
) -> None:
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match=match):
        init_disentangle(xi1, xi2, xi3, epsilon=epsilon, rng=rng)
