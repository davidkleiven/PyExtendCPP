#ifndef PYTHON_OBJECT_LIKE_H
#define PYTHON_OBJECT_LIKE_H

namespace pyextend
{
  class ObjectLike
  {
  public:
    ObjectLike( PyObject *obj ): obj(obj){};
    ObjectLike( const ObjectLike &other ){swap(other);};

    ObjectLike& operator=( const ObjectLike &other )
    {
      if ( this != &other )
      {
        swap(other);
      }
      return *this;
    };


    ~ObjectLike()
    {
      if ( own_ptr )
      {
        Py_DECREF(obj);
      }
    };


    /** This object owns the underlying pointer */
    void set_own(){ own_ptr=true; };



    /** Get a raw pointer to the underlying object */
    PyObject* raw_ptr() const { return obj; };


    /** Increase the reference count */
    void incref(){ Py_INCREF(obj); };


    /** Get the ref count of the underlying object */
    int refcnt() const { return obj->ob_refcnt; };
  protected:
    PyObject *obj;
    bool own_ptr{false};



    virtual void swap( const ObjectLike &other )
    {
      obj = other.obj;
      own_ptr = other.own_ptr;
      if ( other.own_ptr )
      {
        Py_INCREF(obj);
      }
    }
  };
}
#endif
