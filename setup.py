from setuptools import setup

setup(
    name='cumulant',
    version='0.0.1',
    install_requires=[
        'numpy',
        'qutip',
        'scipy; python_version > "3.6"',
    ],
)
