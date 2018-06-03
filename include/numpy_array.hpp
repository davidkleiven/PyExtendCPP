#ifndef NUMPY_ARRAY_WRAPPER_H
#define NUMPY_ARRAY_WRAPPER_H
#include "object_like.hpp"
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL cool_ARRAY_API
#include <numpy/ndarrayobject.h>
#include <vector>

#define npcast(dtype,value) *static_cast<##dtype*>(value)

namespace pyextend
{
  template<class T>
  struct npy_typenum{static const int value=0;};
  template<>
  struct npy_typenum<double>{static const int value=NPY_DOUBLE;};
  template<>
  struct npy_typenum<int>{static const int value=NPY_INT;};
  template<>
  struct npy_typenum<unsigned int>{static const int value=NPY_UINT;};


  template<class dtype>
  class NumpyArray: public ObjectLike
  {
  public:
    NumpyArray( PyObject *obj ):ObjectLike(obj), \
    otf_ptr(PyArray_FROM_OTF(obj, npy_typenum<dtype>::value, NPY_INOUT_ARRAY)){};

    NumpyArray( const std::vector<dtype> &vec );
    NumpyArray( const std::vector< std::vector<dtype> > &mat );
    NumpyArray( const std::vector< std::vector< std::vector<dtype> > > &mat );

    ~NumpyArray()
    {
      Py_DECREF(otf_ptr);
    };


    /** Access operators */
    const dtype& operator()(int i) const {return *static_cast<dtype*>(PyArray_GETPTR1(otf_ptr,i));};
    dtype& operator()(int i){return *static_cast<dtype*>(PyArray_GETPTR1(otf_ptr,i));};

    const dtype& operator()(int i, int j) const {return *static_cast<dtype*>(PyArray_GETPTR2(otf_ptr,i,j));}
    dtype& operator()(int i, int j) {return *static_cast<dtype*>(PyArray_GETPTR2(otf_ptr,i,j));}

    const dtype& operator()(int i, int j, int k) const {return *static_cast<dtype*>(PyArray_GETPTR3(otf_ptr,i,j,k));}
    dtype& operator()(int i, int j, int k) {return *static_cast<dtype*>(PyArray_GETPTR3(otf_ptr,i,j,k));}


    /** Gives the shape of the array */
    void shape( std::vector<int> &shp ) const
    {
      int ndims = PyArray_NDIM(otf_ptr);
      npy_intp* dims = PyArray_DIMS(otf_ptr);
      for (unsigned int i=0;i<ndims;i++ )
      {
        shp.push_back(dims[i]);
      }
    };


    /** Returns the size of the first dimension */
    unsigned int size() const
    {
      int ndims = PyArray_NDIM(otf_ptr);
      npy_intp* dims = PyArray_DIMS(otf_ptr);
      return dims[0];
    };
  private:
    PyObject *otf_ptr{nullptr};


    void swap( const NumpyArray<dtype> &other )
    {
      ObjectLike::swap(other);
      otf_ptr = other.otf_ptr;
      Py_INCREF(otf_ptr);
    };
  };


  // Implement special constructors
  template<class dtype>
  NumpyArray<dtype>::NumpyArray( const std::vector<dtype> &vec ):ObjectLike(nullptr)
  {
    own_ptr = true;
    npy_intp dims[1] = {vec.size()};
    obj = PyArray_SimpleNew( 1, dims, npy_typenum<dtype>::value );
    otf_ptr = obj;

    for (unsigned int i=0;i<vec.size();i++ )
    {
      double* ptr = PyArray_GETPTR1(otf_ptr, i);
      *ptr = vec[i];
    }
  }


  template<class dtype>
  NumpyArray<dtype>::NumpyArray( const std::vector< std::vector<dtype> > &mat):ObjectLike(nullptr)
  {
    own_ptr = true;
    npy_intp dims[2] = {mat.size(), mat[0].size()};
    obj = PyArray_SimpleNew( 2, dims, npy_typenum<dtype>::value );
    otf_ptr = obj;
    for (unsigned int i=0;i<mat.size();i++ )
    {
      for (unsigned int j=0;j<mat.size();j++ )
      {
        double *ptr = PyArray_GETPTR2(otf_ptr, i, j);
        *ptr = mat[i][j];
      }
    }
  }


  template<class dtype>
  NumpyArray<dtype>::NumpyArray( const std::vector< std::vector< std::vector<dtype> > > &mat ): \
  ObjectLike(nullptr)
  {
    own_ptr = true;
    npy_intp dims[3] = {mat.size(), mat[0].size(), mat[0][0].size()};
    obj = PyArray_SimpleNew( 3, dims, npy_typenum<dtype>::value );
    otf_ptr = obj;
    for (unsigned int i=0;i<mat.size();i++ )
    for (unsigned int j=0;j<mat[i].size();j++ )
    for (unsigned int k=0;k<mat[i][j].size();k++ )
    {
      double *ptr = PyArray_GETPTR3(otf_ptr, i, j, k);
      *ptr = mat[i][j][k];
    }
  }
}

#endif
