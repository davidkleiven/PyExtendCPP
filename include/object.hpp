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


    /** Get the size of the object */
    virtual unsigned int size() const
    {
      throw std::invalid_argument("General Python objects have no function size!");
    }


    template<class T>
    virtual T operator[](int indx)
    {
      throw std::invalid_argument("General Python objects have no [] operator");
    }
  private:
    PyObject *obj;
  };
}
#endif
