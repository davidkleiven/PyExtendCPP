#ifndef PYTHON_OBJECT_H
#define PYTHON_OBJECT_H
#include <Python.h>
#include <stdexcept>
#include "object_like.hpp"
#include "ensure_object_like.hpp"
//#include "dtype_converter.hpp"
#include <sstream>

namespace pyextend
{
  template <class T>
  class DataTypeConverter;

  class Object: public ObjectLike
  {
  public:
    Object( PyObject* obj ):ObjectLike(obj){};


    /** Get an attribute from the class */
    template<class dtype>
    dtype attr( const char* name )
    {
      DataTypeConverter<dtype> converter;
      PyObject *pyattr = get_pyattr(name);

      // This version is intended for the case when the python object
      // can be convertd to a double,int etc. so we can DECREF
      // the python pointer
      Py_DECREF(pyattr);
      return converter.py2c(pyattr);
    };


    /** Get attr for object like attributes */
    template<class dtype>
    dtype attr_obj( const char* name )
    {
      ensure_object_like<dtype> typecheck;
      DataTypeConverter<dtype> converter;
      PyObject* pyattr = get_pyattr(name);
      dtype value = converter.py2c(pyattr);
      value.set_own();
      return value;
    };


    /** Set attribute */
    template<class dtype>
    void set_attr( const char* name, const dtype& value )
    {
      DataTypeConverter<dtype> converter;
      PyObject_SetAttr(obj, name, converter.c2py(value));
    };
  private:
    PyObject* get_pyattr( const char *name )
    {
      PyObject* pyattr = PyObject_GetAttrString( obj, name );
      if ( pyattr == NULL )
      {
        std::stringstream ss;
        ss << "Object does not have attribute " << name;
        throw std::invalid_argument(ss.str());
      }
      return pyattr;
    };
  };




};
#endif
