![Build status](https://travis-ci.org/davidkleiven/PyExtendCPP.svg?branch=master)

# PyExtendCPP
**PyExtendCPP** is a C++ interface to the Python C API.
The intention is to simplify the API in Python extension for the most common
data structures including
1. Lists of integers or floats
2. Nested lists of integers or floats
3. NumPy arrays of integers or floats
4. Accessing attributes from objects which is of any of the types above

# Example 1: Manipulating a 2D Numpy Array
If we have a python extension that takes a Numpy array as an argument
we can access the elements of the numpy array as follows
```cpp
#include <pyextend/pyextend.hpp>
using namespace pyextend;
PyObject *ex_manipulate_numpy_array( PyObject* self, PyObject *args )
{
  PyObject* nparray=nullptr;

  // Parse the arguments
  if ( !PyArg_ParseTuple(args, "O", &nparray) )
  {
    // Wrapper for raising erros provided by pyextend
    type_error("Could not parse arguments");
    return NULL;
  }

  NumpyArray<double> array(nparray);

  // Access element 0,1
  double value = array(0,1);

  // Setting element 2,0
  array(2,0) = 10.0;
}
```
