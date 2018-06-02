#ifndef TEST_LIST_H
#define TEST_LIST_H
#include <Python.h>

/** Computes the sum of the elements in a list */
PyObject* sum_list( PyObject* self, PyObject* args );

/** Computest the sum of all elements in a nested list */
PyObject* sum_nested( PyObject* self, PyObject *args );
#endif
