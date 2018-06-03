#ifndef PYTHON_LIST_WRAPPER_H
#define PYTHON_LIST_WRAPPER_H
#include <Python.h>
#include <vector>
#include "dtype_converter.hpp"
#include "object_like.hpp"

namespace pyextend
{
  template<class dtype>
  class List: public ObjectLike
  {
  public:
    List(PyObject *obj):ObjectLike(obj){};


    /** Returns the size of the list */
    unsigned int size() const
    {
      return PyList_Size(obj);
    };

    /** Read operator */
    dtype operator[](int indx)
    {
      return converter.py2c(PyList_GetItem(obj,indx));
    };


    /** Set an item */
    void set( dtype value, int indx )
    {
      PyList_SetItem( obj, indx, converter.c2py(value) );
    };


    /** Converts the list to a C++ vector */
    void to_vector( std::vector<dtype> & vec )
    {
      unsigned int length = size();
      for (unsigned int i=0;i<length;i++ )
      {
        vec.push_back((*this)[i]);
      }
    };
  private:
    pyextend::DataTypeConverter<dtype> converter;
  };
};
#endif
