import numpy as np
import jax.numpy as jnp

class BosonicBath:
    def __init__(self, T):
        self.T = T

    def bose(self, ν):
        r"""
        It computes the Bose-Einstein distribution

        $$ n(\omega)=\frac{1}{e^{\beta \omega}-1} $$

        Parameters:
        ----------
        ν: float
            The mode at which to compute the thermal population

        Returns:
        -------
        float
            The thermal population of mode ν
        """
        if self.T == 0:
            return 0
        if np.isclose(ν,0).all():
            return 0
        return np.exp(-ν / self.T) / (1-np.exp(-ν / self.T))

    def spectral_density(self, w):
        return None

    def correlation_function(self, t):
        return None

    def power_spectrum(self, w):
        return 2*(self.bose(w)+1)*self.spectral_density(w)


class OhmicBath(BosonicBath):
    def __init__(self, T, coupling, cutoff):
        super().__init__(T)
        self.coupling = coupling
        self.cutoff = cutoff
        self.label="ohmic"
    def spectral_density(self, w):
        r"""
        It describes the spectral density of an Ohmic spectral density given by
        
        $$ J(\omega)= \alpha \omega e^{-\frac{|\omega|}{\omega_{c}}} $$
        
        Parameters
        ----------
        """
        return self.coupling*w*np.exp(-abs(w)/self.cutoff)

    def correlation_function(self, t):
        return None


class OverdampedBath(BosonicBath):
    def __init__(self, T, coupling, cutoff):
        super().__init__(T)
        self.coupling = coupling
        self.cutoff = cutoff
        self.label="overdamped"
        self.params=np.array([coupling,cutoff],dtype=np.float32)
    def spectral_density(self, w):
        return 2*self.coupling*self.cutoff*w/(self.cutoff**2 + w**2)

    def vkr(self,k):
        if k==0:
            return self.cutoff
        else:
            return 2*np.pi*k*self.T
    def ckr(self,k):
        c,d=self.coupling,self.cutoff
        if k==0:
            return c*d/np.tan(d/(2*self.T))
        else:
            vk=self.vkr(k)
            return 4*c*d*vk*self.T/(vk**2 - d**2)
    def vki(self,k):
        if k==0:
            return self.cutoff
        else:
            return 0
    def vk(self,k):
        return self.vkr(k)+1j*self.vki(k)
    def cki(k,bath):
        if k==0:
            return -bath.coupling*bath.cutoff
        else: 
            return 0

    def correlation_function(self, t,k=1000):
        return (self.ckr(k)+1j*self.cki(k))*np.exp(-(self.vkr(k)
                                                     +1j*self.vki(k))*t)


class UnderdampedBath(BosonicBath):
    def __init__(self, T):
        super().__init__(T)

    def spectral_density(self, w):
        return super().spectral_density(w)

    def correlation_function(self, t):
        return super().correlation_function(t)
