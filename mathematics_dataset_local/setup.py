from setuptools import setup, find_packages

setup(
    name='mathematics_dataset_local',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        # List the package dependencies here
        'sympy',
        'numpy',
        'six',
    ],
)
