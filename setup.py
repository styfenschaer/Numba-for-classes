from setuptools import find_packages, setup


setup(
    name="numbas",
    version="1.0.0",
    author="Styfen Schär",
    description="Just-in-time compile class methods with Numba",
    packages=find_packages(),
    install_requires=["numba"],
    license="BSD-3-Clause",

)
