#ifndef TEST_NUMPY_ARRAY_H
#define TEST_NUMPY_ARRAY_H
#include <Python.h>

PyObject* sum1D( PyObject* self, PyObject *args );
PyObject* sum2D( PyObject* self, PyObject *args );
PyObject* sum3D( PyObject* self, PyObject *args );
#endif
