import numpy as np
from scipy.integrate import quad_vec
import jax.numpy as jnp
from qutip import spre as qutip_spre
from qutip import spost as qutip_spost
from qutip import Qobj as qutip_Qobj
from nmm.utils.utils import spre as jax_spre
from nmm.utils.utils import spost as jax_spost
from nmm.utils.utils import Qobj as jax_Qobj
import itertools
from collections import defaultdict
from nmm.cumulant.cum import bath_csolve
from multipledispatch import dispatch
import warnings
from scipy.integrate import solve_ivp
from jax import tree_util
from scipy.interpolate import interp1d


@dispatch(qutip_Qobj)
def spre(op):
    return qutip_spre(op)


@dispatch(qutip_Qobj)
def spost(op):
    return qutip_spost(op)


@dispatch(jax_Qobj)
def spre(op):
    return jax_spre(op)


@dispatch(jax_Qobj)
def spost(op):
    return jax_spost(op)


class blochredfield:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, matsubara=True,
                 points=1000):
        self.points = points
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        if isinstance(Hsys, qutip_Qobj):
            self._qutip = True
        else:
            self._qutip = False
        self.baths = baths
        self.Qs = Qs
        self.matsubara = matsubara

    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.eps,
                    self.limit, self.baths, self.dtype)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def _gamma(self, ν, bath, w, w1, t):
        r"""
        It describes the Integrand of the decay rates of the cumulant equation
        for bosonic baths

        $$\Gamma(w,w',t)=\int_{0}^{t} dt_1 \int_{0}^{t} dt_2
        e^{i (w t_1 - w' t_2)} \mathcal{C}(t_{1},t_{2})$$

        Parameters:
        ----------

        w: float or numpy.ndarray

        w1: float or numpy.ndarray

        t: float or numpy.ndarray

        Returns:
        --------
        float or numpy.ndarray
            It returns a value or array describing the decay between the levels
            with energies w and w1 at time t

        """
        try:
            bath.bose(w)
        except:
            bath.bose = bath._bose_einstein
            self._mul = 1/np.pi
        var = 1j*bath.spectral_density(ν)*bath.bose(
            ν)*(1-np.exp(1j*t*(w+ν)))/(w+ν)
        var += 1j*bath.spectral_density(ν)*(
            bath.bose(ν)+1)*(1-np.exp(1j*t*(w-ν)))/(w-ν)
        var2 = 1j*bath.spectral_density(ν)*bath.bose(
            ν)*(1-np.exp(1j*t*(w1+ν)))/(w1+ν)
        var2 += 1j*bath.spectral_density(ν)*(
            bath.bose(ν)+1)*(1-np.exp(1j*t*(w1-ν)))/(w1-ν)
        return (var2 + np.conjugate(var))*self._mul

    def _gamma_gen(self, bath, w, w1, t):
        r"""
        It describes the the decay rates of the cumulant equation
        for bosonic baths

        $$\Gamma(\omega,\omega',t) = t^{2}\int_{0}^{\infty} d\omega 
        e^{i\frac{\omega-\omega'}{2} t} J(\omega) \left[ (n(\omega)+1) 
        sinc\left(\frac{(\omega-\omega)t}{2}\right)
        sinc\left(\frac{(\omega'-\omega)t}{2}\right)+ n(\omega) 
        sinc\left(\frac{(\omega+\omega)t}{2}\right) 
        sinc\left(\frac{(\omega'+\omega)t}{2}\right)   \right]$$

        Parameters
        ----------

        w : float or numpy.ndarray
        w1 : float or numpy.ndarray
        t : float or numpy.ndarray

        Returns
        -------
        float or numpy.ndarray
            It returns a value or array describing the decay between the levels
            with energies w and w1 at time t

        """
        if isinstance(t, type(jnp.array([2]))):
            t = np.array(t.tolist())

        # integrals = quad_vec(
        #     self._gamma,
        #     0,
        #     np.Inf,
        #     args=(bath, w, w1, t),
        #     points=[-w, -w1, w, w1],
        #     epsabs=self.eps,
        #     epsrel=self.eps,
        #     quadrature="gk15"
        # )[0]
        return np.exp(1j*(w-w1)*t)*bath.power_spectrum(w)

    def jump_operators(self, Q):
        evals, all_state = self.Hsys.eigenstates()
        N = len(all_state)
        collapse_list = []
        ws = []
        for j in range(N):
            for k in range(j + 1, N):
                Deltajk = evals[k] - evals[j]
                ws.append(Deltajk)
                collapse_list.append(
                    (
                        all_state[j]
                        * all_state[j].dag()
                        * Q
                        * all_state[k]
                        * all_state[k].dag()
                    )
                )  # emission
                ws.append(-Deltajk)
                collapse_list.append(
                    (
                        all_state[k]
                        * all_state[k].dag()
                        * Q
                        * all_state[j]
                        * all_state[j].dag()
                    )
                )  # absorption
        collapse_list.append(Q - sum(collapse_list))  # Dephasing
        ws.append(0)
        output = defaultdict(list)
        for k, key in enumerate(ws):
            output[np.round(key, 12)].append(collapse_list[k])
        eldict = {x: sum(y) for x, y in output.items()}
        dictrem = {}
        empty = 0*self.Hsys
        for keys, values in eldict.items():
            if not (values == empty):
                dictrem[keys] = values
        return dictrem

    def decays(self, combinations, bath, t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self._gamma_gen(bath, i[0], i[1], t)
        return rates

    def matrix_form(self, jumps, combinations):
        matrixform = {}
        for i in combinations:
            matrixform[i] = (
                spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) - 1 *
                (0.5 *
                 (spre(jumps[i[0]].dag() * jumps[i[1]]) +
                  spost(jumps[i[0]].dag() * jumps[i[1]]))))
        return matrixform

    def prepare_interpolated_generators(self):
        print("Started interpolation")
        try:
            generators = self.generators
        except:
            self.generator()
            generators = np.array([i.full().flatten()
                                  for i in self.generators])

        self.interpolated_generators = [
            interp1d(
                self.t, generators[:, i],
                kind='linear', bounds_error=False,
                fill_value="extrapolate")
            for i in range(generators.shape[1])]
        self.generator_shape = self.generators[0].shape

    def interpolated_generator(self, t):
        if self.interpolated_generators is None:
            raise ValueError(
                "Interpolated generators not prepared. Call prepare_interpolated_generators() first.")

        interpolated = np.array([interp(t)
                                for interp in self.interpolated_generators])
        return interpolated.reshape(self.generator_shape)

    def generator(self):
        generators = []
        for Q, bath in zip(self.Qs, self.baths):
            jumps = self.jump_operators(Q)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            matrices = self.matrix_form(jumps, combinations)
            decays = self.decays(combinations, bath, self.t)
            superop = []
            if self._qutip:
                gen = (np.array(matrices[i])*decays[i] for i in combinations)
            else:
                gen = (matrices[i]*(decays[i]).item() for i in combinations)
            superop.append(sum(gen))
            generators.extend(superop)
            del gen
            del matrices
            del decays
        generate = sum(generators)
        self.generators = generate

    def evolution(self, rho0, method="BDF"):
        r"""
        This function computes the evolution of the state $\rho(0)$

        Parameters
        ----------

        rho0 : numpy.ndarray or qutip.Qobj
            The initial state of the quantum system under consideration.

        approximated : bool
            When False the full cumulant equation/refined weak coupling is
            computed, when True the Filtered Approximation (FA is computed),
            this greatly reduces computational time, at the expense of
            diminishing accuracy particularly for the populations of the system
            at early times.

        Returns
        -------
        list
            a list containing all of the density matrices, at all timesteps of
            the evolution
        """
        # be easility jitted
        # try:
        #     y0=rho0.data.flatten()
        #     y0=np.array(y0).astype(np.complex128)
        #     f=lambda t,y: np.array(self.generator(t).data)@np.array(y) #maybe this can
        # except:
        y0 = rho0.full().flatten()
        y0 = np.array(y0).astype(np.complex128)
        # def f(t, y): return self.generator(
        #     t).full()@np.array(y)  # maybe this can
        self.prepare_interpolated_generators()

        def f(t, y):
            return self.interpolated_generator(t) @ y
        result = solve_ivp(f, [0, self.t[-1]],
                           y0,
                           t_eval=self.t, method=method)
        n = self.Hsys.shape[0]
        states = [result.y[:, i].reshape(n, n)
                  for i in range(len(self.t))]

        return states

    def _decayww2(self, bath, w, w1, t, k=100):
        term1 = (bath.ckr(k)-1j*bath.cki(k)
                 )*(np.exp(1j*t*(w-w1))-np.exp(-t*(bath.vk(k)+1j*w1)))
        term1 = term1/(bath.vk(k)+1j*w)
        term2 = (bath.ckr(k)+1j*bath.cki(k)
                 )*(np.exp(1j*t*(w-w1))-np.exp(-t*(bath.vk(k)-1j*w)))
        term2 = term2/(bath.vk(k)-1j*w1)
        return (term1+term2)*np.pi

    def _decayww(self, bath, w, t, k=100):
        return self._decayww2(bath, w, w, t, k)

    def decayww2(self, bath, w, w1, t, k=100):
        return np.sum(self._decayww2(bath, w, w1, t, k))

    def decayww(self, bath, w, t, k=100):
        return np.sum(self._decayww(bath, w, t, k))


tree_util.register_pytree_node(
    redfield,
    redfield._tree_flatten,
    redfield._tree_unflatten)

# TODO Add Lamb-shift
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO support Tensor Networks
# Benchmark with the QuatumToolbox,jl based version
# TODO catch warning from scipy
# Habilitate double precision (Maybe single is good for now)
