#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#include "caffe/layers/proposal_layer.hpp"
#include <stdio.h>
namespace caffe
{ 

  
template<typename T, typename Dtype>
struct TLess : public thrust::binary_function<T,T,bool>
{
  __host__ __device__ bool operator()(const T &a, const T &b) const {
     return m_values[a]>=m_values[b];
  }
  thrust::device_ptr<Dtype> m_values;
};   
  
  
__device__ int bottom_offset(const int n, const int c,  const int h, const int w,
                                   const int channels, const int height, const int width)
{
  return ((n * channels + c) * height + h) * width + w;
}

template <typename Dtype>
__host__ __device__ Dtype IOU(const Dtype x1_i, const Dtype y1_i, const Dtype x2_i, const Dtype y2_i, 
                     const Dtype x1_j, const Dtype y1_j, const Dtype x2_j, const Dtype y2_j )
{
    Dtype area_j = (x2_j-x1_j+1) * (y2_j-y1_j+1);

    Dtype area_i = (x2_i-x1_i+1) * (y2_i-y1_i+1);

    Dtype  x1_inter = max(x1_j, x1_i);
    Dtype  y1_inter = max(y1_j, y1_i);
    Dtype  x2_inter = min(x2_j, x2_i);
    Dtype  y2_inter = min(y2_j, y2_i);
    Dtype inter = max(0.0, x2_inter - x1_inter + 1) * max(0.0, y2_inter - y1_inter + 1);
    
    return inter / (area_i + area_j - inter);
}

template <typename Dtype>
__global__ void create_full_anchors(const int nthreads,
                                    const Dtype* base_anchors, const int num_anchors, const int width, const int feat_stride,
                                    Dtype* anchors)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int j = index%num_anchors;
    int i = index/num_anchors;
    
    anchors[index*4] = base_anchors[j*4+0]+(i%width)*feat_stride;
    anchors[index*4+1] = base_anchors[j*4+1]+(i/width)*feat_stride;
    anchors[index*4+2] = base_anchors[j*4+2]+(i%width)*feat_stride;
    anchors[index*4+3] = base_anchors[j*4+3]+(i/width)*feat_stride;
  }
}

template <typename Dtype>
__global__ void create_full_proposals(const int nthreads, 
                                      const Dtype* const bottom_data_1 ,const int channels_1, const int height_1, const int width_1 ,
                                      const Dtype* const bottom_data_0, const int channels_0, const int height_0, const int width_0 ,
                                      Dtype* proposals, const Dtype* anchors, Dtype* scores, int* indexes,
                                      const Dtype* const im_info, const int num_anchors , const int min_size)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    Dtype img_w = im_info[1];
    Dtype img_h = im_info[0];
    Dtype minimal_size = ((Dtype)min_size) * im_info[2];
    
    int ch_0 = index % (channels_0-num_anchors) ;
    int w_0 = (index / (channels_0-num_anchors)) % width_0;
    int h_0 = (index / (channels_0-num_anchors)) / width_0;
    
    int ch_1 = index % (channels_1/4);
    int w_1 = (index / (channels_1/4)) % width_1;
    int h_1 = (index / (channels_1/4)) / width_1;

    
    Dtype width = anchors[index*4+2] - anchors[index*4] + (Dtype)1;
    Dtype height = anchors[index*4+3] - anchors[index*4+1] + (Dtype)1;
    Dtype ctr_x = anchors[index*4] + (Dtype)0.5 * width;
    Dtype ctr_y = anchors[index*4+1] + (Dtype)0.5 * height;
            
    Dtype pred_ctr_x = bottom_data_1[bottom_offset(0,ch_1*4, h_1, w_1, channels_1, height_1, width_1)] * width + ctr_x;
    Dtype pred_ctr_y = bottom_data_1[bottom_offset(0,ch_1*4+1, h_1, w_1, channels_1, height_1, width_1)]* height + ctr_y;
    Dtype pred_w = exp( bottom_data_1[bottom_offset(0,ch_1*4+2, h_1, w_1, channels_1, height_1, width_1)] ) * width;
    Dtype pred_h = exp( bottom_data_1[bottom_offset(0,ch_1*4+3, h_1, w_1, channels_1, height_1, width_1)] ) * height;
    
    proposals[index*4] = max(min(pred_ctr_x - (Dtype)0.5 * pred_w, img_w-(Dtype)1),(Dtype)0.0); 
    proposals[index*4+1] = max(min(pred_ctr_y - (Dtype)0.5 * pred_h, img_h-(Dtype)1),(Dtype)0.0); 
    proposals[index*4+2] = max(min(pred_ctr_x + (Dtype)0.5 * pred_w,img_w-(Dtype)1),(Dtype)0.0);
    proposals[index*4+3] = max(min( pred_ctr_y + (Dtype)0.5 * pred_h,img_h-(Dtype)1),(Dtype)0.0);
    
    indexes[index] = index;
    scores[index] = 
      (proposals[index*4+2]-proposals[index*4]+(Dtype)1)>=minimal_size && (proposals[index*4+3]-proposals[index*4+1]+(Dtype)1)>=minimal_size ?
    bottom_data_0[bottom_offset(0,ch_0+num_anchors, h_0, w_0, channels_0, height_0, width_0 )] : -1;
    
  }
}

// template <typename Dtype>
// __global__ void nms_map_kernel(const int indexes_count, const int* indexes, const Dtype* proposals,const Dtype* scores, const Dtype threshold, int* map)
// {
//  
//   int i = (blockIdx.x * blockDim.x) + threadIdx.x;
//   int j = (blockIdx.y * blockDim.y) + threadIdx.y;
//   
// 
//   if ( i < indexes_count && j < indexes_count){
//       int i_ind = indexes[i];
//       int j_ind = indexes[j];
//       //map[i*indexes_count + j]=false;
//       if(scores[i_ind]<scores[j_ind])
//       {
//         
//         Dtype iou = IOU<Dtype>(proposals[i_ind*4],proposals[i_ind*4+1],proposals[i_ind*4+2],proposals[i_ind*4+3],
//                                proposals[j_ind*4],proposals[j_ind*4+1],proposals[j_ind*4+2],proposals[j_ind*4+3]);
//         if(iou>=threshold)
//         {
//           map[i*indexes_count + j]= 1;
//         }
//       }
//   }
// }
// 
// template <typename Dtype>
// __global__ void nms_reduce_kernel(const int indexes_count, const int* indexes, const int* map, int* reduction, Dtype* scores)
// {
//   int i = blockIdx.x;
//   int j = i * indexes_count + threadIdx.x;
//   int n = blockDim.x;
//   
//   reduction[i] =__syncthreads_or(map[ j < ( (i+1) * indexes_count ) ? j : ( (i+1) * indexes_count - 1 )]);
//   
//   for(int t = 1; t <= indexes_count / n + 1; ++t )
//   {
//     j = j + n;
//     reduction[i] = __syncthreads_or(reduction[i] || map[j < ( (i+1) * indexes_count ) ? j : ( (i+1) * indexes_count - 1 )]);
//   }
//   __syncthreads();
//   if(threadIdx.x==0)
//   {
//     scores[indexes[i]]= reduction[i] ? -1.0 : scores[indexes[i]]; 
//   }
// }

template <typename Dtype>
__global__ void nms_reduce_kernel_imp(const int indexes_count, const int* indexes, const Dtype* proposals, Dtype* scores, int* reduction, const Dtype threshold)
{
  int i = blockIdx.x;
  int j = threadIdx.x;
  int n = blockDim.x;
  
  int i_ind = indexes[i];
  int j_ind = indexes[j <  indexes_count ? j : indexes_count-1];

  reduction[i]=__syncthreads_or( scores[i_ind]<scores[j_ind] && 
                                 IOU<Dtype>(proposals[i_ind*4],proposals[i_ind*4+1],proposals[i_ind*4+2],proposals[i_ind*4+3],
                                 proposals[j_ind*4],proposals[j_ind*4+1],proposals[j_ind*4+2],proposals[j_ind*4+3]) >= threshold );
  
  for(int t=1; t<=indexes_count/n+2; ++t )
  {
    j= j+ n;
    j_ind = indexes[j < indexes_count ? j : indexes_count-1];
     
    reduction[i]=__syncthreads_or(reduction[i] || (scores[i_ind]<scores[j_ind] && 
                                 IOU<Dtype>(proposals[i_ind*4],proposals[i_ind*4+1],proposals[i_ind*4+2],proposals[i_ind*4+3],
                                 proposals[j_ind*4],proposals[j_ind*4+1],proposals[j_ind*4+2],proposals[j_ind*4+3]) >= threshold));
  }
  __syncthreads();
  if(threadIdx.x==0)
  {
    scores[indexes[i]]= reduction[i] ? -1.0 : scores[indexes[i]]; 
  }
}


template <typename Dtype>
__global__ void data_to_top(const int nthreads, const int* indexes, const Dtype* proposals, Dtype* top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads)
  {
    int ind = indexes[index];
    top_data[index*5]=0;
    top_data[index*5+1] = proposals[ind*4];
    top_data[index*5+2] = proposals[ind*4+1];
    top_data[index*5+3] = proposals[ind*4+2];
    top_data[index*5+4] = proposals[ind*4+3];
  }
}

template <typename Dtype>
void nms_cpu_ref(const int indexes_count, const int* indexes, Dtype* scores, Dtype* proposals ,const Dtype threshold)
{
  std::vector<bool> supressed(indexes_count, false);
  for(int i=0; i<indexes_count; ++i)
  {
      if(supressed[i]) continue;
      for(int j = i+1; j<indexes_count; ++j)
      {
          if(supressed[j])continue;
          int i_ind=indexes[i];
          int j_ind=indexes[j];
          
          
          Dtype overlap = IOU<Dtype>(proposals[i_ind*4],proposals[i_ind*4+1],proposals[i_ind*4+2],proposals[i_ind*4+3],
                              proposals[j_ind*4],proposals[j_ind*4+1],proposals[j_ind*4+2],proposals[j_ind*4+3]);
          supressed[j]= overlap>=threshold;
      }
  }
  for(int i=0; i<indexes_count; ++i)
  {
    scores[indexes[i]]= supressed[i] ? -1.0 : scores[indexes[i]];
  }
  
}
  
template <typename Dtype>
void ProposalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  if(bottom.size()==4)
  {
    m_pre_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,0)];
    m_post_nms_topN = bottom[3]->cpu_data()[bottom[3]->offset(0,1)];
    m_nms_thresh =bottom[3]->cpu_data()[bottom[3]->offset(0,2)];
    m_min_size = bottom[3]->cpu_data()[bottom[3]->offset(0,3)];
  }
  
  int count = m_anchors.count()/4;
  
  create_full_anchors<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    m_base_anchors.gpu_data(), m_num_anchors, bottom[0]->shape(3), m_feat_stride,
    m_anchors.mutable_gpu_data());

  const Dtype* bottom_1 = bottom[1]->gpu_data();
  const Dtype* bottom_0 = bottom[0]->gpu_data();
  
  Dtype* proposals = m_proposals.mutable_gpu_data(); 
  Dtype* scores = m_scores.mutable_gpu_data();
  Dtype* anchors = m_anchors.mutable_gpu_data();
  int* indexes = m_indexes.mutable_gpu_data();
  int* ioumat = m_iou.mutable_gpu_data();
  
  
  create_full_proposals<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
    bottom_1, bottom[1]->shape(1), bottom[1]->shape(2), bottom[1]->shape(3),
    bottom_0, bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3),
    proposals, anchors, scores, indexes,
    bottom[2]->gpu_data(), m_num_anchors, m_min_size);
  CUDA_POST_KERNEL_CHECK;

  
  {
    thrust::device_ptr<Dtype> t_scores(scores);
    thrust::device_ptr<int> t_indexes(indexes);
    TLess<int,Dtype> less;
    less.m_values=t_scores;

    thrust::stable_sort(t_indexes,t_indexes+count,less); 
  }
  
  
  int pre_nms_topN = std::min<int>(m_pre_nms_topN, count);
    
  
//  nms_cpu_ref<Dtype>(pre_nms_topN, m_indexes.cpu_data(), m_scores.mutable_cpu_data(), m_proposals.mutable_cpu_data(), m_nms_thresh);

   
    nms_reduce_kernel_imp<Dtype><<<pre_nms_topN,CAFFE_CUDA_NUM_THREADS>>>(pre_nms_topN,indexes,proposals, scores, m_reduce.mutable_gpu_data(), m_nms_thresh);
  
  {
    thrust::device_ptr<Dtype> t_scores(m_scores.mutable_gpu_data());
    thrust::device_ptr<int> t_indexes(m_indexes.mutable_gpu_data());
    TLess<int,Dtype> less;
    less.m_values=t_scores;

    thrust::stable_sort(t_indexes,t_indexes+pre_nms_topN,less); 
  }

  data_to_top<Dtype><<<CAFFE_GET_BLOCKS(m_post_nms_topN), CAFFE_CUDA_NUM_THREADS>>>(m_post_nms_topN, indexes, proposals, top[0]->mutable_gpu_data() );
  CUDA_POST_KERNEL_CHECK;

}


template <typename Dtype>
void ProposalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
   return;
}

INSTANTIATE_LAYER_GPU_FUNCS(ProposalLayer);
}
