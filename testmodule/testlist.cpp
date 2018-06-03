#include "testlist.hpp"
#include <pyextend/pyextend.hpp>
#include <cmath>
#include <vector>
#include <iostream>

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
  list_int.set(0,1);
  list_dbl.set(0,1.0);


  // Test to vector
  vector<double> vec;
  list_dbl.to_vector(vec);

  return PyFloat_FromDouble(int_sum);
}


PyObject* sum_nested( PyObject *self, PyObject *args )
{
  PyObject *nested_list;
  if ( !PyArg_ParseTuple(args, "O", &nested_list) )
  {
    type_error("Could not parse arguments");
    return NULL;
  }

  double sum=0.0;
  List< List<double> > list(nested_list);
  for ( unsigned int i=0;i<list.size();i++ )
  {
    List<double> sublist = list[i];
    for ( unsigned int j=0;j<sublist.size();j++ )
    {
      sum += sublist[j];
    }
  }
  return PyFloat_FromDouble(sum);
}


PyObject* list_from_vector( PyObject *self, PyObject *args )
{
  vector<int> vec;
  vec.push_back(0);
  vec.push_back(1);
  vec.push_back(2);
  vec.push_back(3);

  List<int> list(vec);

  // We return the object created in this function so we have to
  // increase the reference count
  list.incref();
  return list.raw_ptr();
}
