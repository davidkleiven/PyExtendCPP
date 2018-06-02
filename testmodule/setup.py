from setuptools import setup, Extension

module = Extension("testmodule_cpp", sources=["testmodule.cpp","testlist.cpp"], \
language="c++")

setup(
    name = "pyextend_testmodule",
    ext_modules=[module]
)
