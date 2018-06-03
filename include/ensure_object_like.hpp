#ifndef ENSURE_OBJECT_LIKE_H
#define ENSURE_OBJECT_LIKE_H

namespace pyextend
{
  class ObjectLike;
  template<class dtype>
  struct is_object_like
  {
    static const bool value=false;
  };

  template<>
  struct is_object_like<ObjectLike>
  {
    static const bool value=true;
  };

  template<class T>
  struct ensure_object_like
  {
    static_assert(is_object_like<T>::value, "Requires an ObjectLike data type!");
  };
}
#endif
