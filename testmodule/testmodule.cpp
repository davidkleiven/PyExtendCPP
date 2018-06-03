#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL PYEXTEND_ARRAY_API
#include <numpy/ndarrayobject.h>

#include "testlist.hpp"
#include "testobject.hpp"
#include "testnumpy.hpp"

static PyMethodDef testmodule_methods[] = {
  {"sum_list", sum_list, METH_VARARGS, "Sum the entries of a list"},
  {"sum_nested", sum_nested, METH_VARARGS, "Sum all entreis in a nested list"},
  {"test_access", test_access, METH_VARARGS, "Test the read/write functions of objects"},
  {"list_from_vector", list_from_vector, METH_VARARGS, "Create a list from a vector"},
  {"sum1D", sum1D, METH_VARARGS, "Sum all entries in an 1D numpy array"},
  {"sum2D", sum2D, METH_VARARGS, "Sum all entries in an 2D numpy array"},
  {"sum3D", sum3D, METH_VARARGS, "Sum all entries in an 3D numpy array"},
  {"create1D", create1D, METH_VARARGS, "Create a 1D array from a C++ vector"},
  {"create2D", create2D, METH_VARARGS, "Create a 2D array from a C++ vector"},
  {"create3D", create3D, METH_VARARGS, "Create a 3D array from a C++ vector"},
  {NULL,NULL,0,NULL}
};

#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef testmodule = {
    PyModuleDef_HEAD_INIT,
    "testmodule_cpp",
    NULL, // TODO: Write documentation string here
    -1,
    testmodule_methods
  };
#endif

#if PY_MAJOR_VERSION >= 3
  PyMODINIT_FUNC PyInit_testmodule_cpp(void)
  {
    PyObject* module = PyModule_Create( &testmodule );
    import_array();
    return module;
  };
#else
  PyMODINIT_FUNC inittestmodule_cpp(void)
  {
    Py_InitModule3( "testmodule_cpp", testmodule_methods, "This the Python 2 version" );
    import_array();
  };
#endif
