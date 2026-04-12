import numpy as np

from dlam_mc_core.kernels import dreaming_core


def test_dreaming_core_at_zero_sleep_is_identity() -> None:
    c = np.array([[1.2, 0.3], [0.3, 0.9]], dtype=np.float64)
    out = dreaming_core(c, t=0.0)
    assert np.allclose(out, np.eye(2))


def test_dreaming_core_output_shape() -> None:
    c = np.array([[0.8, 0.1, 0.0], [0.1, 1.0, 0.2], [0.0, 0.2, 0.7]], dtype=np.float64)
    out = dreaming_core(c, t=2.0)
    assert out.shape == (3, 3)
