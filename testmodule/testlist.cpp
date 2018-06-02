#include "testlist.hpp"
#include <pyextend/pyextend.hpp>
#include <cmath>
#include <vector>

using namespace pyextend;
using namespace std;

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

  // Try the set methods
  list_int.set(1,0);
  list_dbl.set(1.0,0);


  // Test to vector
  vector<double> vec;
  list_dbl.to_vector(vec);

  return PyInt_FromLong(int_sum);
}
