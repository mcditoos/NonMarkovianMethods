import numpy as np
from scipy.integrate import quad_vec
from tqdm.notebook import tqdm
from qutip import spre, spost, Qobj
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
            t
            * t
            * np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w - ν) / (2 * np.pi) * t)
               * np.sinc((w1 - ν) / (2 * np.pi) * t))
            * (self.bose(ν) + 1)
        )
        var += (
            t
            * t
            * np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w + ν) / (2 * np.pi) * t)
               * np.sinc((w1 + ν) / (2 * np.pi) * t))
            * self.bose(ν)
        )
        return var
    def γdot(self, ν, w, w1, t):
        var = ((1/(w-ν))*(1/(ν-w1))
            * np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w - ν) / (2 * np.pi) * t)
               * np.sinc((w1 - ν) / (2 * np.pi) * t))
            * (self.bose(ν) + 1)
        )
        var += (
            t
            * t
            * np.exp(1j * (w - w1) / 2 * t)
            * self.spectral_density(ν)
            * (np.sinc((w + ν) / (2 * np.pi) * t)
               * np.sinc((w1 + ν) / (2 * np.pi) * t))
            * self.bose(ν)
        )
        return var

    def Γgen(self, w, w1, t, regularized=False):
        if regularized:
            return self.γ_star(w, w1, t)
        return quad_vec(
            lambda ν: self.γ(ν, w, w1, t),
            0,
            np.Inf,
            epsabs=self.eps,
            epsrel=self.eps,
            quadrature="gk15",
        )[0]
    def Γgendot(self, w, w1, t, regularized=False):
        return quad_vec(
            lambda ν: self.γdot(ν, w, w1, t),
            0,
            np.Inf,
            epsabs=self.eps,
            epsrel=self.eps,
            quadrature="gk15",
        )[0]

    def generator(self, regularized=False):
        """Generates the cumulant super operator,can be made faster by looking
        at the upper triangular only (half the integrals)"""
        superop = 0
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
        for keys, values in eldict.items():
            if values != Qobj([[0]*N]*N):
                dictrem[keys] = values
        ws = dictrem.keys()
        eldict = dictrem
        combinations = list(itertools.product(ws, ws))
        decays = []
        matrixform = []
        for i in tqdm(combinations, desc='generating generators ...'):
            decays.append(self.Γgen(i[0], i[1], self.t, regularized))
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
# TODO pictures
# TODO better naming
# TODO explain regularization issues
