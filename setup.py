from setuptools import setup,find_packages

setup(
    name='nmm',
    version='0.0.1',
    install_requires=[
        'numpy',
        'scipy; python_version > "3.6"',
    ],
    packages=find_packages(),
    extras_require={"Full": ['qutip']},
)
