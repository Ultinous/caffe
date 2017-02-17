#ifndef CAFFE_UTIL_NMS_H_
#define CAFFE_UTIL_NMS_H_

#include "caffe/util/device_alternate.hpp"

namespace caffe
{
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const int threadsPerBlock =  sizeof(unsigned long long) * 8;

template <typename Dtype>
Dtype host_IOU(Dtype const * const a, Dtype const * const b);

template <typename Dtype>
int nms_cpu(const int& boxes_num, int* indexes, const Dtype* scores,const Dtype* proposals , const Dtype& threshold);

#ifndef CPU_ONLY
template <typename Dtype>
int nms_gpu(const int& boxes_num, int* indexes, const Dtype* scores, const Dtype* proposals ,const Dtype& threshold);

#endif
  
}

#endif