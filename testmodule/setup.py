from setuptools import setup, Extension
import numpy as np

module = Extension("testmodule_cpp", sources=["testmodule.cpp","testlist.cpp",\
"testobject.cpp", "testnumpy.cpp"], \
language="c++", include_dirs=[np.get_include()], extra_compile_args=["-std=c++11"])

setup(
    name = "pyextend_testmodule",
    ext_modules=[module]
)
