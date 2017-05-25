#ifndef CAFFE_UTIL_GPU_UTIL_H_
#define CAFFE_UTIL_GPU_UTIL_H_

namespace caffe {

template <typename Dtype>
inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

template <>
inline __device__
float caffe_gpu_atomic_add(const float val, float* address) {
  return atomicAdd(address, val);
}

// double atomicAdd implementation taken from:
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
template <>
inline __device__
double caffe_gpu_atomic_add(const double val, double* address) {
  unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
      // NOLINT_NEXT_LINE(runtime/int)
      reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
  unsigned long long int assumed;  // NOLINT(runtime/int)
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}



//Ulti code
template<typename Dtype>
inline __device__
Dtype dev_IoU(Dtype const * const a, Dtype const * const b)
{
Dtype area_a = (a[2]-a[0]+1) * (a[3]-a[1]+1);
Dtype area_b = (b[2]-b[0]+1) * (b[3]-b[1]+1);

Dtype inter_x1 = max(a[0], b[0]);
Dtype inter_y1 = max(a[1], b[1]);
Dtype inter_x2 = min(a[2], b[2]);
Dtype inter_y2 = min(a[3], b[3]);
Dtype inter = max((Dtype)0, inter_x2 - inter_x1 + 1) * max((Dtype)0, inter_y2 - inter_y1 + 1);

return inter / (area_a + area_b - inter);
}

}  // namespace caffe

#endif  // CAFFE_UTIL_GPU_UTIL_H_
