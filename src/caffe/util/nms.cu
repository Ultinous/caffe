#include <vector>

#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "caffe/util/nms.hpp"

namespace caffe {


struct thrust_is_zero
{
  __device__
  bool operator()(const int x) const {
    return ( !x ); }
};

template<typename Dtype>
__device__ inline Dtype dev_IoU(Dtype const * const a, Dtype const * const b)
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


template<typename Dtype>
__global__ void nms_kernel(const int boxes_num, const int* indexes,
                           const Dtype *boxes, unsigned long long *dev_mask,
                           const Dtype nms_overlap_thresh ) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  // if (row_start > col_start) return;

  const int row_size =
          min(boxes_num - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
          min(boxes_num - col_start * threadsPerBlock, threadsPerBlock);
  __shared__ Dtype block_boxes[threadsPerBlock*4];
  if (threadIdx.x < col_size) {
    int index = indexes[(threadsPerBlock * col_start + threadIdx.x)];
    block_boxes[threadIdx.x * 4 + 0] = boxes[index * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] = boxes[index * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] = boxes[index * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] = boxes[index * 4 + 3];

  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const Dtype *cur_box = boxes + indexes[cur_box_idx] * 4;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (dev_IoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}


template<typename Dtype>
int nms_gpu(const int& boxes_num, int* indexes, const Dtype* scores, const Dtype* proposals ,const Dtype& threshold) {
  if (boxes_num <=1) return boxes_num;
  unsigned long long *mask_dev = NULL;

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
              DIVUP(boxes_num, threadsPerBlock));
  dim3 threads(threadsPerBlock);

  nms_kernel < Dtype ><<<blocks, threads>>>(boxes_num, indexes,
          proposals, mask_dev,
          threshold);
  CUDA_POST_KERNEL_CHECK;

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  std::vector<int> keep_out(boxes_num, 0);
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      keep_out[i] =  1;
      ++num_to_keep;
      unsigned long long *p = &mask_host[0] + i * col_blocks;

      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }

  {
    thrust::device_ptr<int> dev_ptr = thrust::device_pointer_cast(indexes);
    thrust::device_vector<int> stencil(keep_out.begin(), keep_out.end());
    thrust::remove_if(dev_ptr,dev_ptr+boxes_num,stencil.begin(), thrust_is_zero());
  }

  CUDA_CHECK(cudaFree(mask_dev));
  return num_to_keep;
}
template int nms_gpu<double>(const int& boxes_num, int* indexes, const double* scores, const double* proposals ,const double& threshold);

template int nms_gpu<float>(const int& boxes_num, int* indexes, const float* scores, const float* proposals ,const float& threshold);

}//caffe