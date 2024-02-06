cimport numpy as np

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
