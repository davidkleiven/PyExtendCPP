#ifndef ENSURE_OBJECT_LIKE_H
#define ENSURE_OBJECT_LIKE_H

namespace pyextend
{
  class ObjectLike;
  class Object;
  template<class T>
  class List;
  template<class T>
  class NumpyArray;


  template<class dtype>
  struct is_object_like
  {
    static const bool value=false;
  };


  template<>
  struct is_object_like<ObjectLike>{static const bool value=true;};
  template<>
  struct is_object_like<Object>{static const bool value=true;};
  template<class T>
  struct is_object_like< List<T> >{static const bool value=true;};
  template<class T>
  struct is_object_like< NumpyArray<T> >{static const bool value=true;};


  template<class T>
  struct ensure_object_like
  {
    static_assert(is_object_like<T>::value, "Requires an ObjectLike data type!");
  };
}
#endif
