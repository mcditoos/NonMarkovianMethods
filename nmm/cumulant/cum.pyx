from cython.cimports.libc.math cimport sin,cos,exp,sqrt,pi,INFINITY
import cython
from scipy.integrate import quad
import numpy as np

cdef sinc(x:cython.float):
    if x==0:
        return 1
    return sin(pi*x)/(pi*x)

cdef extern from "complex.h":
    float complex I

cdef class csolve:
    cdef cython.float[:] t 
    cdef cython.float T
    cdef public cython.float eps 
    cdef cython.float alpha 
    cdef cython.float wc
    def __init__(self, t, T,eps,alpha,wc):
        self.t = t
        self.T=T
        self.eps=eps
        self.alpha=alpha
        self.wc=wc

    cpdef spectral_density(self, w:cython.float):
        return 2*w*self.alpha*self.wc/(self.wc**2 + w**2)
    cpdef bose(self, nu):
        if self.T == 0.0:
            return 0.0
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
        # This doesn't give any significant speed up over just using quad in an inhereited python class
        return [quad(self._gamma,0,INFINITY,args=(w, w1, i),epsabs=self.eps,epsrel=self.eps,complex_func=True)[0] for i in t]

