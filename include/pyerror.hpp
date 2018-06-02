#ifndef PY_ERROR_H
#define PY_ERROR_H
#include <Python.h>

namespace pyextend
{
  /** Type Error */
  void error( PyObject* type, const char* msg )
  {
    PyErr_SetString( type, msg );
  };


  /** Set type erro */
  void type_error( const char* msg )
  {
    error( PyExc_TypeError, msg );
  };


  /** Set value error */
  void value_error( const char* msg )
  {
    error( PyExc_ValueError, msg );
  };
}
#endif
