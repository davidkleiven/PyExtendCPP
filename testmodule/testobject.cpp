#include "testobject.hpp"
#include <pyextend/pyextend.hpp>

using namespace pyextend;

PyObject* test_access( PyObject *self, PyObject *args )
{
  PyObject *obj=nullptr;
  if ( !PyArg_ParseTuple( args, "O", &obj) )
  {
    type_error("Test access could not parse arguments!");
    return nullptr;
  }

  pyextend::Object c_obj(obj);

  // Get double
  double dbl_attr = c_obj.attr<double>( "dbl_attr" );
  int int_attr = c_obj.attr<int> ( "int_attr" );
  List<int> list_attr = c_obj.attr_obj< List<int> >( "list_attr" );

  // Try to set the attributes
  c_obj.set_attr( "dbl_attr", 10.0 );
  c_obj.set_attr( "int_attr", 2 );
  list_attr.set( 0, 10 );
  c_obj.set_attr( "list_attr", list_attr );
  Py_RETURN_TRUE;
}
