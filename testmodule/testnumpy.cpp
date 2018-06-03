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


PyObject* create1D( PyObject *self, PyObject *args )
{
  vector<double> vec;
  vec.push_back(1.0);
  vec.push_back(2.0);
  vec.push_back(3.0);
  NumpyArray<double> data(vec);
  data.incref();
  return data.raw_ptr();
}


PyObject* create2D( PyObject *self, PyObject *args )
{
  vector< vector<double> > vec;
  for (unsigned int i=0;i<5;i++ )
  {
    vector<double> subvec;
    subvec.push_back(1.0);
    subvec.push_back(2.0);
    subvec.push_back(3.0);
    vec.push_back(subvec);
  }

  NumpyArray<double> data(vec);
  data.incref();
  return data.raw_ptr();
}


PyObject* create3D( PyObject *self, PyObject *args )
{
  vector< vector<vector<double> > > vec;
  for (unsigned int i=0;i<5;i++ )
  {
    vector< vector<double> > subvec;
    for (unsigned int j=0;j<4;j++ )
    {
      vector<double> subsubvec;
      subsubvec.push_back(1.0);
      subsubvec.push_back(2.0);
      subsubvec.push_back(3.0);
      subvec.push_back(subsubvec);
    }
    vec.push_back(subvec);
  }

  NumpyArray<double> data(vec);
  data.incref();
  return data.raw_ptr();
}
