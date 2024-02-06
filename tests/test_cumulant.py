import pytest
import numpy as np
from cumulant import csolve
from qutip import sigmax, sigmaz


@pytest.fixture
def init():
    Hsys = sigmaz()/2
    Q = sigmaz()
    lam = 0.05
    gamma = 5
    eps = 1e-6
    t = np.linspace(0, 1000, 10)
    T = 1
    obj = csolve(Hsys, t, eps, lam, gamma, T, Q, 'ohmic')
    return obj


class TestCumulant:
    def test_bose(self, init):
        assert np.isclose(2*init.bose(1)+1, 1/np.tanh(1/(2)))
        assert np.isclose(init.bose(0), 0)

    def test_spectral_density(self, init):
        assert np.isclose(init.spectral_density(0), 0)
        assert np.isclose(init.spectral_density(1e8), 0)
