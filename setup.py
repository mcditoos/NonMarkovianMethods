from setuptools import setup

setup(
    name='cumulant',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy; python_version > "3.6"',
    ],
    extras_require={"Full": ['qutip']},
)
