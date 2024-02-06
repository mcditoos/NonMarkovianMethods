import numpy as np


class spre:
    def __init__(self, op):
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

    def __sub__(self, other):
        self.right -= other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self

    def __mul__(self, other):
        if type(other) in (int, float, complex):
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
        if type(other) in (int, float, complex):
            self.right *= other
            self.func = lambda x: (
                self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
            return self
        self.right = self.right @ other.right
        self.func = lambda x: (
            self.right@x.reshape(self.dim**2)).reshape(self.dim, self.dim)
        return self
