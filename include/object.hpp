#ifndef PYTHON_OBJECT_H
#define PYTHON_OBJECT_H
#include <Python.h>
#include <stdexcept>

namespace pyextend
{
  class Object
  {
  public:
    Object( PyObject* obj ):obj(obj){};

    /** Get a raw pointer to the under lying object */
    PyObject* raw_ptr(){return obj;};

  private:
    PyObject *obj;
  };
};
#endif
