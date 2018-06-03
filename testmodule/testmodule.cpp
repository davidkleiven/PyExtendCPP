#include <Python.h>
#include <numpy/ndarrayobject.h>

#include "testlist.hpp"
#include "testobject.hpp"

static PyMethodDef testmodule_methods[] = {
  {"sum_list", sum_list, METH_VARARGS, "Sum the entries of a list"},
  {"sum_nested", sum_nested, METH_VARARGS, "Sum all entreis in a nested list"},
  {"test_access", test_access, METH_VARARGS, "Test the read/write functions of objects"},
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
