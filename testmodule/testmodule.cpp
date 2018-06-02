#include <Python.h>
#include <numpy/ndarrayobject.h>

#include "testlist.hpp"

static PyMethodDef testmodule_methods[] = {
  {"sum_list", sum_list, METH_VARARGS, "Sum the entries of a list"},
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
  PyMODINIT_FUNC PyInit_testmodule(void)
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
