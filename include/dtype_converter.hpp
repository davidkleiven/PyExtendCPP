#ifndef DTYPE_CONVERTER_H
#define DTYPE_CONVERTER_H
#include <stdexcept>
#include "object.hpp"

namespace pyextend
{
  template<class T>
  class DataTypeConverter
  {
  public:
    DataTypeConverter(){};


    /** Converts a data type to python */
    PyObject* c2py( const T& value )
    {
      throw std::invalid_argument("Cannot convert arbitrary types to Python objects!");
    };

    /** Convert python object to C++ type */
    Object py2c( PyObject* pyobj )
    {
      return Object(pyobj);
    };
  };



  /** PyObject to double */
  template <>
  class DataTypeConverter<double>
  {
  public:
    DataTypeConverter(){};

    /** Conert C-type to python */
    PyObject* c2py( double value )
    {
      return PyFloat_FromDouble(value);
    };

    double py2c( PyObject* pyobj )
    {
      return PyFloat_AsDouble(pyobj);
    };
  };



  /** Int object */
  template<>
  class DataTypeConverter<int>
  {
  public:
    DataTypeConverter(){};

    /** Convert C-type to python */
    PyObject* c2py( int value )
    {
      return PyInt_FromLong(value);
    };

    int py2c( PyObject *pyobj )
    {
      return PyInt_AsLong(pyobj);
    };
  };
};
#endif
