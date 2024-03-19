from setuptools import setup,find_packages
from setuptools import Extension,setup
from Cython.Build import cythonize

extensions = [Extension("cumulant_cy", ["cumulant/_cumulant.pyx"])]

setup(
    name='nmm',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy; python_version > "3.6"',
    ],
    packages=find_packages(),
    extras_require={"Full": ['qutip']},
    ext_modules=cythonize(extensions),

)
