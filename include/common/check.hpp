
#define hipCheck(err)                                     \
  {                                                       \
    hipError_t _err = (err);                              \
    if (_err != hipSuccess)                               \
      printf("HIP error: %s\n", hipGetErrorString(_err)); \
  }

#define HIP_CHECK(err) hipCheck(err)
