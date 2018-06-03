#include "testnumpy.hpp"
#include <pyextend/pyextend.hpp>
#include <iostream>

using namespace pyextend;
using namespace std;

PyObject *sum1D( PyObject *self, PyObject *args )
{
  PyObject *array=nullptr;
  if ( !PyArg_ParseTuple(args, "O", &array) )
  {
    type_error("sum1D could not parse arguments!");
    return NULL;
  }

  NumpyArray<double> npy_array(array);
  double sum = 0.0;
  for (unsigned int i=0;i<npy_array.size();i++ )
  {
    sum += npy_array(i);
  }
  return PyFloat_FromDouble(sum);
}


PyObject *sum2D( PyObject *self, PyObject *args )
{
  PyObject *array=nullptr;
  if ( !PyArg_ParseTuple(args, "O", &array) )
  {
    type_error("sum1D could not parse arguments!");
    return NULL;
  }

  NumpyArray<double> npy_array(array);
  double sum = 0.0;
  vector<int> shape;
  npy_array.shape(shape);
  for (unsigned int i=0;i<shape[0];i++ )
  for (unsigned int j=0;j<shape[1];j++ )
  {
    sum += npy_array(i,j);
  }
  return PyFloat_FromDouble(sum);
}


PyObject *sum3D( PyObject *self, PyObject *args )
{
  PyObject *array=nullptr;
  if ( !PyArg_ParseTuple(args, "O", &array) )
  {
    type_error("sum1D could not parse arguments!");
    return NULL;
  }

  NumpyArray<double> npy_array(array);
  double sum = 0.0;
  vector<int> shape;
  npy_array.shape(shape);
  for (unsigned int i=0;i<shape[0];i++ )
  for (unsigned int j=0;j<shape[1];j++ )
  for (unsigned int k=0;k<shape[2];k++ )
  {
    sum += npy_array(i,j,k);
  }
  return PyFloat_FromDouble(sum);
}
