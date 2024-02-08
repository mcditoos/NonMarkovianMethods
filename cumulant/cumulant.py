import numpy as np
from scipy.integrate import quad_vec
from tqdm.notebook import tqdm
try:
    from qutip import spre, spost, Qobj
    _qutip = True
except ModuleNotFoundError:
    _qutip = False
    from utils import spre, spost
import itertools


class csolve:
    def __init__(self, Hsys, t, eps, lam, gamma, T, Q, typeof, w0=1):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        self.lam = lam
        self.gamma = gamma
        self.T = T
        self.Q = Q
        self.typeof = typeof
        self.w0 = w0

    def bose(self, ν):
        if self.T == 0:
            return 0
        if ν == 0:
            return 0
        return 1 / (np.exp(ν / self.T) - 1)

    def spectral_density(self, w):
        if self.typeof == "ohmic":
            return self.lam * w * np.exp(-abs(w) / self.gamma)
        elif self.typeof == 'ud':
            return self.lam**2 * self.gamma * w / ((w**2 - self.w0**2)**2
                                                   + (self.gamma*w)**2)
        elif self.typeof == 'od':
            return 2 * w * self.lam*self.gamma / (self.gamma**2 + w**2)
        else:
            return None

    def γ_star(self, w, w1, t):
        """
        One start regularization
        """
        var = (2 * np.pi * t * np.exp(1j * (w1 - w) * t / 2)
               * np.sinc((w1 - w) * t / (2 * np.pi))
               * np.sqrt(self.spectral_density(w1) * (self.bose(w1) + 1))
               * np.sqrt(self.spectral_density(w) * (self.bose(w) + 1)))
        return var

    def γ(self, ν, w, w1, t):
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

    def Γgen(self, w, w1, t, regularized=False):
        if regularized:
            return self.γ_star(w, w1, t)
        else:
            integrals = quad_vec(
                self.γ,
                0,
                np.Inf,
                args=(w, w1, t),
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk15",
                workers=-1
            )[0]
            return t*t*integrals

    def generator(self, regularized=False):
        """Generates the cumulant super operator"""
        superop = 0
        if type(self.Hsys) != np.ndarray:
            evals, all_state = self.Hsys.eigenstates()
        else:
            evals, all_state = np.linalg.eig(self.Hsys)

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
        collapse_list.append(self.Q - sum(collapse_list))  # Dephasing
        ws.append(0)
        eldict = {ws[i]: collapse_list[i] for i in range(len(ws))}
        dictrem = {}
        if _qutip:
            empty = Qobj([[0]*N]*N)
        else:
            empty = ([[0]*N]*N)
        for keys, values in eldict.items():
            if values != empty:
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
                rates[i] = self.Γgen(i[0], i[1], self.t, regularized)

        for i in tqdm(combinations, desc='Calculating the generator ...'):
            decays.append(rates[i])
            if _qutip is False:
                matrixform.append(
                    (spre(eldict[i[1]]) @ spost(eldict[i[0]].dag()) -
                     (
                        (spre(eldict[i[0]].dag() @ eldict[i[1]]) + spost(
                            eldict[i[0]].dag() @ eldict[i[1]])*0.5))))
            else:
                matrixform.append(
                    (spre(eldict[i[1]]) * spost(eldict[i[0]].dag()) -
                     (0.5 *
                     (spre(eldict[i[0]].dag() * eldict[i[1]]) + spost(
                         eldict[i[0]].dag() * eldict[i[1]])))))
        ll = []
        superop = []
        for l in range(len(self.t)):
            ll.append([decays[j][l] * matrixform[j]
                       for j in range(len(combinations))])
            superop.append(sum(ll[0]))
            ll = []
        self.generators = superop

    def evolution(self, rho0, regularized=False):
        self.generator(regularized)
        return [i.expm()(rho0) for i in tqdm(self.generators,
                desc='Computing Exponential of Generators . . . .')]


# TODO Add Lamb-shift


    def ξ(self, ν, w, w0, T, t, wc, lam):
        if ν == w:
            return 0
        if ν == -w:
            return 0
        return (
            t
            * t
            * (1 / (4 * np.pi))
            * (
                np.sinc((w - w0) * t / (2 * np.pi)) ** 2
                - np.sinc((w0 + w) * t / (2 * np.pi)) ** 2
            )
            * self.spectral_density(ν, lam, wc)
            * ((self.bose(ν, T) + 1) / (w - ν) + (self.bose(ν, T)) / (w + ν))
        )
# TODO pictures
# TODO better naming
# TODO explain regularization issues
