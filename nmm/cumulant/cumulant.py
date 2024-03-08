import numpy as np
from scipy.integrate import quad_vec
from tqdm import tqdm
try:
    from qutip import spre, spost, Qobj
    _qutip = True
except ModuleNotFoundError:
    _qutip = False
    from nmm.utils.utils import spre, spost
    from scipy.linalg import expm
import itertools


class csolve:
    def __init__(self, Hsys, t, bath, Q, eps=1e-4):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.bose = bath.bose
        self.spectral_density = bath.spectral_density
        self.Q = Q

    def γfa(self, w, w1, t):
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
               * np.sqrt(self.spectral_density(w1) * (self.bose(w1) + 1))
               * np.sqrt(self.spectral_density(w) * (self.bose(w) + 1)))
        return var

    def _γ(self, ν, w, w1, t):
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
        var = (
            np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w - ν) / (2 * np.pi) * t)
               * np.sinc((w1 - ν) / (2 * np.pi) * t))
            * (self.bose(ν) + 1)
        )
        var += (
            np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w + ν) / (2 * np.pi) * t)
               * np.sinc((w1 + ν) / (2 * np.pi) * t))
            * self.bose(ν)
        )
        return var

    def Γgen(self, w, w1, t, approximated=False):
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
        if approximated:
            return self.γfa(w, w1, t)
        else:
            integrals = quad_vec(
                self._γ,
                0,
                np.Inf,
                args=(w, w1, t),
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk15"
            )[0]
            return t*t*integrals

    def generator(self, approximated=False):
        superop = 0
        if type(self.Hsys) != np.ndarray:
            evals, all_state = self.Hsys.eigenstates()
        else:
            evals, all_state = np.linalg.eig(self.Hsys)
            all_state = [i.reshape((len(i), 1)) for i in all_state]

        N = len(all_state)
        collapse_list = []
        ws = []
        for j in range(N):
            for k in range(j + 1, N):
                Deltajk = evals[k] - evals[j]
                ws.append(Deltajk)
                if type(self.Hsys) != np.ndarray:
                    collapse_list.append(
                        (
                            all_state[j]
                            * all_state[j].dag()
                            * self.Q
                            * all_state[k]
                            * all_state[k].dag()
                        )
                    )  # emission
                    ws.append(-Deltajk)
                    collapse_list.append(
                        (
                            all_state[k]
                            * all_state[k].dag()
                            * self.Q
                            * all_state[j]
                            * all_state[j].dag()
                        )
                    )  # absorption
                else:
                    collapse_list.append(
                        (
                            all_state[j]
                            @ np.conjugate(all_state[j]).T
                            @ self.Q
                            @ all_state[k]
                            @ np.conjugate(all_state[k]).T
                        )
                    )  # emission
                    ws.append(-Deltajk)
                    collapse_list.append(
                        (
                            all_state[k]
                            @ np.conjugate(all_state[k]).T
                            @ self.Q
                            @ all_state[j]
                            @ np.conjugate(all_state[j]).T
                        )
                    )  # absorption
        collapse_list.append(self.Q - sum(collapse_list))  # Dephasing
        ws.append(0)
        eldict = {ws[i]: collapse_list[i] for i in range(len(ws))}
        dictrem = {}
        if _qutip:
            empty = Qobj([[0]*N]*N)
            for keys, values in eldict.items():
                if (values != empty):
                    dictrem[keys] = values
        else:
            empty = np.array([[0]*N]*N)
            for keys, values in eldict.items():
                if (values != empty).any():
                    dictrem[keys] = values
        ws = list(dictrem.keys())
        eldict = dictrem
        combinations = list(itertools.product(ws, ws))
        decays = []
        matrixform = []
        rates = {}
        done = []
        for i in tqdm(combinations, desc='Calculating Integrals ...'):
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.Γgen(i[0], i[1], self.t, approximated)

        for i in tqdm(combinations, desc='Calculating the generator ...'):
            decays.append(rates[i])
            if _qutip is False:
                matrixform.append(
                    (spre(eldict[i[1]]) * spost(
                        np.conjugate(eldict[i[0]]).T) -
                     ((spre(np.conjugate(eldict[i[0]]).T @ eldict[i[1]]) +
                       spost(np.conjugate(eldict[i[0]]).T @ eldict[i[1]])
                       )*0.5)))
            else:
                matrixform.append(
                    (spre(eldict[i[1]]) * spost(eldict[i[0]].dag()) -
                     (0.5 *
                     (spre(eldict[i[0]].dag() * eldict[i[1]]) + spost(
                         eldict[i[0]].dag() * eldict[i[1]])))))
        ll = []
        superop = []
        for l in range(len(self.t)):
            if _qutip:
                ll = [matrixform[j]*decays[j][l]
                      for j in range(len(combinations))]
                superop.append(sum(ll))
            else:
                ll = [matrixform[j].right*decays[j][l]
                      for j in range(len(combinations))]
                superop.append(sum(ll))
            ll = []
        self.generators = superop

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
        if _qutip:
            states=[(i).expm()(rho0) for i in tqdm(self.generators,
                    desc='Computing Exponential of Generators . . . .')]
        else:
            states=[(expm(i)@(rho0.reshape(rho0.shape[0]**2)))
                    .reshape(rho0.shape)
                    for i in tqdm(self.generators,
                    desc='Computing Exponential of Generators . . . .')]
        return [(-1j*self.Hsys*self.t[i]).expm()*states[i]*
                (1j*self.Hsys*self.t[i]).expm() for i in range(len(self.t))]


# TODO Add Lamb-shift
# TODO Measure times // Mix with numba/cython
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO Add support to for multiple baths (it can be done simply by calling
# generators on the different baths and adding)
