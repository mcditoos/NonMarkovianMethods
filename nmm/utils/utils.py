import numpy as np
from scipy.linalg import expm


class spre:
    def __init__(self, op):
        op = op.astype('complex128')
        self.right = np.kron(op, np.eye(op.shape[0]))
        self.dim = op.shape[0]
        self.func = lambda x: (
            self.right@x.reshape(op.shape[0]**2)).reshape(op.shape[0], op.shape[0])

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        self.right += other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __radd__(self, other):
        if type(other) == int:
            self.right += other
            return self
        self.right += other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __sub__(self, other):
        self.right -= other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __mul__(self, other):
        if type(other) in (int, float, complex, np.complex128):
            self.right *= other
            self.func = lambda x: (
                self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
            return self
        self.right = self.right @ other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self


class spost:
    def __init__(self, op):
        op = op.astype('complex128')
        self.right = np.kron(np.eye(op.shape[0]), op.T)
        self.dim = op.shape[0]
        self.func = lambda x: (
            self.right@x.reshape(op.shape[0]**2)).reshape(op.shape[0], op.shape[0])

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        self.right += other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __sub__(self, other):
        self.right -= other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __mul__(self, other):
        if type(other) in (int, float, complex, np.complex128):
            self.right *= other
            self.func = lambda x: (
                self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
            return self
        self.right = self.right @ other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self
    
class result:
    def __init__(self,rho,tlist,time,malloc):
        self.states=rho
        self.time=tlist
        self.duration=time
        self.malloc=malloc
        
