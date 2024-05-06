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


class redfield:
    def __init__(self, Hsys, t, baths,Qs, eps=1e-4):
        self.Hsys = Hsys
        self.t = t
        self.eps = eps
        if isinstance(Hsys,qutip_Qobj):
            self._qutip=True
        else:
            self._qutip=False
        self.baths=baths
        self.Qs = Qs
  
    def _γ(self, ν,bath, w,w1, t):
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
        var = 1j*bath.spectral_density(ν)*bath.bose(ν)*(1-np.exp(1j*t*(w+ν)))/(w+ν)
        var+= 1j*bath.spectral_density(ν)*(bath.bose(ν)+1)*(1-np.exp(1j*t*(w-ν)))/(w-ν)
        var2 = 1j*bath.spectral_density(ν)*bath.bose(ν)*(1-np.exp(1j*t*(w1+ν)))/(w1+ν)
        var2+=1j*bath.spectral_density(ν)*(bath.bose(ν)+1)*(1-np.exp(1j*t*(w1-ν)))/(w1-ν)
        return var2 + np.conjugate(var)

    def Γgen(self, bath ,w, w1, t):
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
        if isinstance(t,type(jnp.array([2]))):
            t=np.array(t.tolist())


        integrals = quad_vec(
                self._γ,
                0,
                np.Inf,
                args=(bath,w,w1, t),
                points=[-w,-w1,w,w1],
                epsabs=self.eps,
                epsrel=self.eps,
                quadrature="gk15"
            )[0]
        return np.exp(1j*(w-w1)*t)*integrals
    
    def jump_operators(self,Q):
        evals, all_state = self.Hsys.eigenstates()
        N=len(all_state)
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
        for k,key in enumerate(ws):
            output[np.round(key,12)].append(collapse_list[k])
        eldict={x:sum(y) for x, y in output.items()}
        dictrem = {}
        empty =0*self.Hsys
        for keys, values in eldict.items():
            if not (values == empty):
                dictrem[keys] = values
        return dictrem
        
    def decays(self,combinations,bath,t):
        rates = {}
        done = []
        for i in combinations:
            done.append(i)
            j = (i[1], i[0])
            if (j in done) & (i != j):
                rates[i] = np.conjugate(rates[j])
            else:
                rates[i] = self.Γgen(bath,i[0], i[1], t)
        return rates
    
    def matrix_form(self,jumps,combinations):   
        matrixform={}       
        for i in combinations:
                matrixform[i]= (spre(jumps[i[1]]) * spost(jumps[i[0]].dag()) 
                                -1*(0.5 *(spre(jumps[i[0]].dag() * jumps[i[1]]) 
                               + spost(jumps[i[0]].dag() * jumps[i[1]]))))
        return matrixform
    
    def generator(self,t):
        if t==0:
            return (spre(self.Qs[0])*0)
        generators=[]
        for Q,bath in zip(self.Qs,self.baths):
            jumps=self.jump_operators(Q)
            ws=list(jumps.keys())
            combinations=list(itertools.product(ws, ws))
            matrices=self.matrix_form(jumps,combinations)
            decays=self.decays(combinations,bath,t)
            superop=[]
            if self._qutip:
                gen = (matrices[i]*decays[i] for i in combinations)
            else:
                gen = (matrices[i]*(decays[i]).item() for i in combinations)
            superop.append(sum(gen))
            generators.extend(superop)
            del gen
            del matrices
            del decays
        return sum(generators)
    
    def evolution(self,rho0):
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
        #be easility jitted
        try:
            y0=rho0.data.flatten()
            y0=np.array(y0).astype(np.complex128)
            f=lambda t,y: np.array(self.generator(t).data)@np.array(y) #maybe this can

        except:
            y0=rho0.full().flatten()
            y0=np.array(y0).astype(np.complex128)
            f=lambda t,y: self.generator(t).full()@np.array(y) #maybe this can

            
        result = solve_ivp(f, [0, self.t[-1]],
                   y0,
                   t_eval=self.t,method="BDF",verbose=True)
        return result
# TODO Add Lamb-shift
# TODO pictures
# TODO better naming
# TODO explain regularization issues
# TODO make result object
# TODO support Tensor Networks
# Benchmark with the QuatumToolbox,jl based version
# TODO catch warning from scipy 
# Habilitate double precision (Maybe single is good for now)