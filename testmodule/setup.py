from setuptools import setup, Extension
import numpy as np

module = Extension("testmodule_cpp", sources=["testmodule.cpp","testlist.cpp"], \
language="c++", include_dirs=[np.get_include()])

setup(
    name = "pyextend_testmodule",
    ext_modules=[module]
)
