from cython.cimports.libc.math cimport sin,cos,exp,sqrt,pi,INFINITY
import cython
from scipy.integrate import quadpack

cdef sinc(x:cython.float):
    if x==0:
        return 1
    return sin(pi*x)/(pi*x)

cdef extern from "complex.h":
    float complex I

cdef class csolve:
    cpdef __init__(self, T:cython.float,alpha:cython.float,wc:cython.float,t:cython.float, eps:cython.float=1e-4):
        self.t = t
        self.eps = eps
        self.T=T
        self.alpha=alpha
        self.wc=wc
    cpdef spectral_density(self, w:cython.float):
        return 2*w*self.alpha*self.wc/(self.wc**2 + w**2)
    cpdef bose(self, nu):
        if self.T == 0:
            return 0
        return exp(-nu / self.T) / (1-exp(-nu / self.T))
    cpdef gammafa(self, w:cython.float, w1:cython.float, t:cython.float):
        var = (2 * pi * t *(cos((w - w1) / 2 * t) + I * sin((w - w1) / 2 * t))            
               * sinc((w1 - w) * t / 2 )
               * sqrt(self.spectral_density(w1) * (self.bose(w1) + 1))
               * sqrt(self.spectral_density(w) * (self.bose(w) + 1)))
        return var
    cpdef _gamma(self, nu:cython.float, w:cython.float, w1:cython.float, t:cython.float):
        expc=cos((w - w1) / 2 * t) + I *sin((w - w1) / 2 * t)         
        var = t*t*(expc
           * self.spectral_density(nu)
            * (sinc((w - nu) / (2*pi ) * t)
               * sinc((w1 - nu) / (2*pi) * t))
            * (self.bose(nu) + 1)
        )
        var += t*t*(expc
            * self.spectral_density(nu)
            * (sinc((w + nu) / (2*pi ) * t)
               * sinc((w1 + nu) / (2*pi) * t))
            * self.bose(nu)
        )
        return var
    cpdef gamma(self, w:cython.float, w1:cython.float, t):
        return [quad(self._gamma,0,INFINITY,args=(w, w1, i),epsabs=self.eps,epsrel=self.eps,complex_func=True)[0] for i in t]

    cpdef _gammare(self, nu:cython.float, w:cython.float, w1:cython.float, t:cython.float):
        return self._gamma(nu,w,w1,t).real
