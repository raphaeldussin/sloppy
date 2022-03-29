""" setup for sloppy """
import setuptools

exec(open("sloppy/version.py").read())

setuptools.setup(version=__version__)