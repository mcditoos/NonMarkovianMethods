import jax.numpy as jnp
from jax.scipy.linalg import expm
from numbers import Number

class Qobj:
    def __init__(self,op):
        self.data=op
        self.shape=op.shape
        self.dtype=op.dtype

    def __eq__(self,other):
        return jnp.array_equal(self.data,other.data).item()

    def __add__(self,other):
        if isinstance(other,Qobj):
            return Qobj(self.data+other.data)
        if other==0:
            return self
        else:
            raise TypeError(f"Not Implemented for {type(other)}")
  
    def __radd__(self,other):
        return self.__add__(other)

    def __sub__(self,other):
        if isinstance(other,Qobj):
            return Qobj(self.data-other.data)
        if other==0:
            return self
        else:
            raise TypeError(f"Not Implemented for {type(other)}")
    def __rsub__(self,other):
            return -1*self.__sub__(other)
    def dag(self):
        return Qobj(jnp.conjugate(jnp.transpose(self.data)))
    def __str__(self):
        s=f"Operator: \n {self.data}"
        return s
    def __repr__(self):
        return self.__str__()
  
    def eigenstates(self):
        eigvals,eigvecs=jnp.linalg.eigh(self.data)
        eigvals=eigvals.tolist()
        eigvecs = [i.reshape((len(i), 1)) for i in eigvecs]
        eigvecs=[Qobj(i) for i in eigvecs]
        return eigvals,eigvecs
   
    def __mul__(self,other):
        if (isinstance(other,Number)):
            return Qobj(self.data*other)
        return Qobj(self.data @ other.data)
    def __truediv__(self,other):
        if (isinstance(other,Number)):
            return Qobj(self.data/other)
        else:
            raise NotImplementedError("Ill defined Operation")
            
    def __rmul__(self,other):
        if (isinstance(other,Number)):
            return Qobj(self.data * other)    
    def expm(self):
        return Qobj(expm(self.data))
class spre:
    def __init__(self, op,kron=True):
        if kron:
            op=op.data
            self.data = jnp.kron(op, jnp.eye(op.shape[0]))
            self.dim = op.shape[0]
        else:
            self.data=op
            self.dim = int(op.shape[0]**0.5)
        ## it may be worth using tensor contractions here (einsum)
        self.func = lambda x: Qobj((self.data@x.data.reshape(self.dim**2)).reshape(self.dim, self.dim))
    def __eq__(self,other):
        return jnp.array_equal(self.data,other.data).item()
    def __str__(self):
        s=f"SuperOperator: \n {self.data}"
        return s
    def __repr__(self):
        return self.__str__()
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spre(data,kron=False)

    def __radd__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spre(data,kron=False)
        
    def __sub__(self, other):
        if isinstance(other,(spre,spost)):
            data =self.data - other.data
            return spre(data,kron=False)
        else:
            raise TypeError

    def __mul__(self, other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spre(data,kron=False)

    def __rmul__(self,other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spre(data,kron=False)

    def __truediv__(self,other):
        if (isinstance(other,Number)):
            data=self.data/other
            return spost(data,kron=False)
        else:
            raise NotImplementedError("Ill defined Operation")
    def expm(self):
        return spre(expm(self.data),kron=False)
class spost:
    def __init__(self, op,kron=True):
        if kron:
            op=op.data
            self.data = jnp.kron(jnp.eye(op.shape[0]), op.T)
            self.dim = op.shape[0]
        else:
            self.data=op
            self.dim = int(op.shape[0]**0.5)
        ## it may be worth using tensor contractions here (einsum)
        self.func = lambda x: Qobj((self.data@x.data.reshape(self.dim**2)).reshape(self.dim, self.dim))
    def __eq__(self,other):
        return jnp.array_equal(self.data,other.data).item()
    def __str__(self):
        s=f"SuperOperator: \n {self.data}"
        return s
    def __repr__(self):
        return self.__str__()
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spost(data,kron=False)

    def __radd__(self, other):
        if isinstance(other,Number):
            if other==0:
                return self
        data = self.data+other.data
        return spost(data,kron=False)
        
    def __sub__(self, other):
        if isinstance(other,(spre,spost)):
            data =self.data - other.data
            return spost(data,kron=False)
        else:
            raise TypeError

    def __truediv__(self,other):
        if (isinstance(other,Number)):
            data=self.data/other
            return spost(data,kron=False)
        else:
            raise NotImplementedError("Ill defined Operation")
    def __mul__(self, other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spost(data,kron=False)

    def __rmul__(self,other):
        if type(other) in (int, float, complex, jnp.complex128):
            data =self.data* other
            return spost(data,kron=False)
    def expm(self):
        return spost(expm(self.data),kron=False)