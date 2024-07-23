import numpy as np
from scipy.integrate import quad_vec
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm
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
from jax import tree_util


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


class csolve:
    def __init__(self, Hsys, t, baths, Qs, eps=1e-4, cython=True, limit=50,
                 matsubara=False, ps=0):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.limit = limit
        self.dtype = Hsys.dtype
        self.ps = ps

        if isinstance(Hsys, qutip_Qobj):
            self._qutip = True
        else:
            self._qutip = False
        if cython:
            self.baths = [bath_csolve(b.T, eps, b.coupling, b.cutoff, b.label)
                          for b in baths]
        else:
            self.baths = baths
        self.Qs = Qs
        self.cython = cython
        self.matsubara = matsubara

    def _tree_flatten(self):
        children = (self.Hsys, self.t, self.eps,
                    self.limit, self.baths, self.dtype)
        aux_data = {}
        return (children, aux_data)

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

    def gamma_fa(self, bath, w, w1, t):
        r"""
        It describes the decay rates for the Filtered Approximation of the
        cumulant equation

        $$\gamma(\omega,\omega^\prime,t)= 2\pi t e^{i \frac{\omega^\prime
        -\omega}{2}t}\mathrm{sinc} \left(\frac{\omega^\prime-\omega}{2}t\right)
         \left(J(\omega^\prime) (n(\omega^\prime)+1)J(\omega) (n(\omega)+1)
         \right)^{\frac{1}{2}}$$

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
        var = (2 * np.pi * t * np.exp(1j * (w1 - w) * t / 2)
               * np.sinc((w1 - w) * t / (2 * np.pi))
               * np.sqrt(bath.spectral_density(w1) * (bath.bose(w1) + 1))
               * np.sqrt(bath.spectral_density(w) * (bath.bose(w) + 1)))
        return var

    def _gamma_(self, ν, bath, w, w1, t):
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
        var = (
            np.exp(1j * (w - w1) / 2 * t)
            * bath.spectral_density(ν)
            * (np.sinc((w - ν) / (2 * np.pi) * t)
               * np.sinc((w1 - ν) / (2 * np.pi) * t))
            * (bath.bose(ν) + 1)
        )
        var += (
            np.exp(1j * (w - w1) / 2 * t)
            * bath.spectral_density(ν)
            * (np.sinc((w + ν) / (2 * np.pi) * t)
               * np.sinc((w1 + ν) / (2 * np.pi) * t))
            * bath.bose(ν)
        )
        try:
            if self._mul:
                var = var*self._mul
        except:
            pass
        return var

    def gamma_gen(self, bath, w, w1, t, approximated=False):
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
        if approximated:
            return self.gamma_fa(bath, w, w1, t)
        if self.matsubara:
            if w == w1:
                return self.decayww(bath, w, t)
            else:
                return self.decayww2(bath, w, w1, t)
        if self.cython:
            return bath.gamma(np.real(w), np.real(w1), t, limit=self.limit)

        else:
            integrals = quad_vec(
                self._gamma_,
                0,
                np.Inf,
                args=(bath, w, w1, t),
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk15"
            )[0]
            return t*t*integrals

    def sparsify(self, vectors, tol=10):
        dims = vectors[0].dims
        new = []
        for vector in vectors:
            top = np.max(np.abs(vector.full()))
            vector = (vector/top).full().round(tol)*top
            vector = qutip_Qobj(vector).to("CSR")
            vector.dims = dims

            new.append(vector)
        return new

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
            output[jnp.round(key, 12).item()].append(collapse_list[k])
        eldict = {x: self.sparsify([sum(y)])[0] for x, y in output.items()}
        dictrem = {}
        empty = 0*self.Hsys
        for keys, values in eldict.items():
            if not (values == empty):
                dictrem[keys] = values
        return dictrem

    def decays(self, combinations, bath, approximated):
        rates = {}
        done = []
        for i in tqdm(combinations, desc='Calculating Integrals ...',
                      dynamic_ncols=True):
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.gamma_gen(bath, i[0], i[1], self.t,
                                          approximated)
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

    def generator(self, approximated=False):
        generators = []
        for Q, bath in zip(self.Qs, self.baths):
            jumps = self.jump_operators(Q)
            ws = list(jumps.keys())
            combinations = list(itertools.product(ws, ws))
            rates = self.decays(combinations, bath, approximated)
            matrices = self.matrix_form(jumps, combinations)
            superop = sum([rates[i] * np.array(matrices[i])
                           for i in combinations])
            generators.extend(superop)
        self.generators = self._reformat(generators)

    def _reformat(self, generators):
        if len(generators) == len(self.t):
            return generators
        else:
            one_list_for_each_bath = [
                generators
                [i * len(self.t): (i + 1) * len(self.t)]
                for i in range(
                    0, int(
                        len(generators) / len(self.t)))]
            composed = list(map(sum, zip(*one_list_for_each_bath)))
            return composed

    def evolution(self, rho0, approximated=False):
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
        self.generator(approximated)
        states = [
            i.expm()(rho0)
            for i in tqdm(
                self.generators,
                desc='Computing Exponential of Generators . . . .')]  # this counts time incorrectly
        return states

    def _decayww(self, bath, w, t):
        k = self.matsubara
        decay1 = -(1j*bath.ckr(k)+bath.cki(k)
                   )*(1j-1j*np.exp(-t*(1j*w+bath.vk(k)))+t*(w-1j*bath.vk(k)))
        decay1 = decay1/(w-1j*bath.vk(k))**2
        return 2*np.real(decay1)*np.pi

    def _decayww2(self, bath, w, w1, t):
        k = self.matsubara
        mul1 = (1j*bath.ckr(k)+bath.cki(k))/(w1-1j*bath.vk(k))
        tem1 = (np.exp(1j*t*(w-w1))-np.exp(-bath.vk(k)
                * t-1j*w1*t))/(bath.vk(k)+1j*w)
        tem2 = 1j*(np.exp(1j*(w-w1)*t)-1)/(w-w1)
        first = mul1*(tem1+tem2)
        mul2 = -(bath.ckr(k)+1j*bath.cki(k))*(1j*bath.vk(k)+(w-w1)
                                              * np.exp(-bath.vk(k)*t+1j*t*w) - (w+1j*bath.vk(k))*np.exp(1j*t*(w-w1))+w1)
        div2 = (w+1j*bath.vk(k))*(w-w1)*(w1+1j*bath.vk(k))
        secod = mul2/div2
        return (first+secod)*np.pi

    def decayww2(self, bath, w, w1, t):
        return np.array([np.sum(self._decayww2(bath, w, w1, i)) for i in t])

    def decayww(self, bath, w, t):
        return np.array([np.sum(self._decayww(bath, w, i)) for i in t])


tree_util.register_pytree_node(
    csolve,
    csolve._tree_flatten,
    csolve._tree_unflatten)
# TODO Add Lamb-shift
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO support Tensor Networks
# Benchmark with the QuatumToolbox,jl based version
# Habilitate double precision (Maybe single is good for now)
# TODO Diffrax does not work unless one makes a pytree for Qobj apparently
