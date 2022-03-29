""" setup for sloppy """
import setuptools


INSTALL_REQUIRES = ["numpy", "xarray", "dask", "netCDF4", "numba"]
TESTS_REQUIRE = ['pytest >= 2.8', 'pytest_datafiles']

setuptools.setup(
    name="sloppy",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("Build topography file for MOM6 ocean model"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/sloppy",
    packages=["sloppy"],
    install_requires=INSTALL_REQUIRES,
    tests_require=TESTS_REQUIRE,
    scripts=[],
)
