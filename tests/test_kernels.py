import numpy as np
import pytest

from dlam_mc_core.kernels import dreaming_core


def test_dreaming_core_at_zero_sleep_is_identity() -> None:
    c = np.array([[1.2, 0.3], [0.3, 0.9]], dtype=np.float64)
    out = dreaming_core(c, t=0.0)
    assert np.allclose(out, np.eye(2))


def test_dreaming_core_output_shape() -> None:
    c = np.array([[0.8, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 0.7]], dtype=np.float64)
    out = dreaming_core(c, t=2.0)
    assert out.shape == (3, 3)


def test_dreaming_core_matches_closed_form_on_diagonal_case() -> None:
    c = np.diag(np.array([0.5, 1.5, 2.0], dtype=np.float64))
    t = 3.0
    out = dreaming_core(c, t=t)

    expected_diag = (1.0 + t) / (1.0 + t * np.diag(c))
    assert np.allclose(np.diag(out), expected_diag)
    assert np.allclose(out, np.diag(np.diag(out)))


def test_dreaming_core_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="square"):
        dreaming_core(np.ones((2, 3), dtype=np.float64), t=1.0)

    with pytest.raises(ValueError, match=">= 0"):
        dreaming_core(np.eye(2, dtype=np.float64), t=-1e-3)
