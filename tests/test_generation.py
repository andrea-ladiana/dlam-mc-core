import numpy as np

from dlam_mc_core.generation import generate_layer


def test_generate_layer_shapes_and_binary_values() -> None:
    rng = np.random.default_rng(0)
    layer = generate_layer(K=5, N=40, M=7, r=0.7, rng=rng)

    assert layer.Xi.shape == (5, 40)
    assert layer.Eta.shape == (5, 7, 40)
    assert layer.U.shape == (5, 40)
    assert layer.hat_xi.shape == (5, 40)
    assert layer.C.shape == (5, 5)

    assert set(np.unique(layer.Xi)).issubset({-1, 1})
    assert set(np.unique(layer.Eta)).issubset({-1, 1})


def test_generate_layer_rho_definition() -> None:
    rng = np.random.default_rng(1)
    r = 0.65
    m = 9
    layer = generate_layer(K=3, N=20, M=m, r=r, rng=rng)

    expected_rho = (1.0 - r * r) / (m * r * r)
    assert np.isclose(layer.rho, expected_rho)
