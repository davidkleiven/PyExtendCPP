#ifndef PYTHON_LIST_WRAPPER_H
#define PYTHON_LIST_WRAPPER_H
#include <Python.h>
#include <vector>
#include "dtype_converter.hpp"

namespace pyextend
{
  template<class dtype>
  class List
  {
  public:
    List(PyObject *obj):obj(obj), own_raw_ptr(false){};
    List(PyObject *obj, bool own):obj(obj), own_raw_ptr(own){};


   ~List()
   {
     if ( own_raw_ptr )
     {
       Py_DECREF(obj);
     }
   }

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


    /** Get a raw pointer to the underlying object */
    PyObject* raw_ptr() { return obj; };


    /** This object owns the underlying pointer */
    void set_own(){ own_raw_ptr=true; };
  private:
    pyextend::DataTypeConverter<dtype> converter;
    PyObject *obj;
    bool own_raw_ptr{false};
  };
};
#endif
