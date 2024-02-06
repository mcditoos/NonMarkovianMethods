from setuptools import setup
from Cython.Build import cythonize

setup(
    name='cumulant',
    version='0.0.1',
    install_requires=[
        'numpy',
        'qutip',
        'scipy; python_version > "3.6"',
    ],
    ext_modules=cythonize("integrals.pyx")
)
