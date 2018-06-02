#include "testlist.hpp"
#include <pyextend/pyextend.hpp>
#include <cmath>

using namespace pyextend;

PyObject* sum_list( PyObject* self, PyObject* args )
{
  PyObject *list = nullptr;
  if ( !PyArg_ParseTuple(args, "O", &list) )
  {
    type_error("Could not parse arguments");
    return NULL;
  }

  List<int> list_int(list);
  List<double> list_dbl(list);

  int int_sum = 0;
  double dbl_sum = 0.0;
  for ( unsigned int i=0;i<list_int.size();i++ )
  {
    int_sum += list_int[i];
    dbl_sum += list_dbl[i];
  }

  return PyInt_FromLong(int_sum);
}
