import pytest
import jax.numpy as jnp
import numpy as np
import nmm
    
    

@pytest.fixture
def init():
    Hsys = jnp.array([[1, 0], [0, -1]])/2
    Q = jnp.array([[1, 0], [0, -1]])
    alpha= 0.05
    gamma = 5
    t = np.linspace(0, 1000, 10)
    T=1
    bath=nmm.OhmicBath(T,alpha,gamma)
    obj = nmm.cumulant.csolve(Hsys,t ,[bath], [Q])

    return obj


class TestCumulant:
    @pytest.mark.parametrize("ars,expected",
                             [((1, 2, 3), (0.0628+0.8859j)),
                              ((1, 2, 0), 0 + 0j),
                              ((1.05, 0.095, 2), (0.3599 - 0.5087j))])
    def test_γfa(self, init, ars, expected):
        assert np.isclose(init.γfa(init.baths[0],*ars), expected, atol=1e-3)

    @pytest.mark.parametrize("ars,expected",
                             [((1, 2, 3, 4), (0.0023 + 0.0051j)),
                              ((0, 0, 0, 5), 0 + 0j),
                              ((1.05, 0.095, 2, 1), (0.0433-0.0610j))])
    def test_γ(self, init, ars, expected):
        assert np.isclose(init._γ(ars[0],init.baths[0],*ars[1:]), expected, atol=1e-3)

    @pytest.mark.parametrize("ars,expected",
                             [((1, 1, [2]), (0.7490)),
                              ((1, -1, [2]), (-0.1119+0.2445j)),
                              ((-1, -1, [2]), (0.3618))])
    def test_Γgen(self, init, ars, expected):
        assert np.isclose(init.Γgen(init.baths[0],*ars), expected, atol=1e-3).all()


# TODO REDESIGN TEST